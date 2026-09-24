"""SemIf cross-encoder on MLX.

The Qwen3.5 backbone comes from mlx-lm (Metal kernels for gated delta net); this
file adds back the 3-class NLI head that mlx-lm's causal-LM wrapper drops.

Loads either
  * an original Hugging Face checkpoint directory (AlexWortega/openjev/<subdir>), or
  * a pre-converted MLX checkpoint (see tools/convert_to_mlx.py), or
  * a Hub repo id of a pre-converted checkpoint, e.g. "jacksodj/semif-4b-nli-v5-mlx".
"""
import json
import pathlib

import mlx.core as mx
import mlx.nn as nn                       # note: mlx.nn, not torch.nn
import numpy as np
from mlx.utils import tree_unflatten
from mlx_lm.models import qwen3_5
from transformers import AutoTokenizer

LABELS = ["contradiction", "entailment", "neutral"]   # the head's output order
CON, ENT, NEU = 0, 1, 2


class SemIfMLX(nn.Module):
    """Qwen3.5 backbone plus the 3-class head mlx-lm's causal-LM wrapper drops."""

    def __init__(self, config: dict):
        super().__init__()
        self.backbone = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(config))
        self.score = nn.Linear(config["text_config"]["hidden_size"], 3, bias=False)

    def __call__(self, input_ids: mx.array, lengths: mx.array) -> mx.array:
        # .language_model.model is Qwen3_5TextModel, whose __call__ returns
        # self.norm(hidden_states) -- exactly HF's `last_hidden_state`.
        # (mlx-lm 0.31.3 has no Model.model property, hence the long path.)
        hidden = self.backbone.language_model.model(input_ids)
        pooled = hidden[mx.arange(hidden.shape[0]), lengths - 1]   # last non-pad token
        return self.score(pooled)


def resolve(path_or_repo) -> pathlib.Path:
    """A local directory, or a Hub repo id to download."""
    p = pathlib.Path(path_or_repo)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return pathlib.Path(snapshot_download(str(path_or_repo)))


def load_params(model: SemIfMLX, path: pathlib.Path) -> dict:
    """Read safetensors shards and return params keyed for SemIfMLX."""
    weights = {}
    for shard in sorted(path.glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    if any(k.startswith("backbone.") for k in weights):
        return weights                                   # already converted
    score_w = weights.pop("score.weight")                # pull the head out BEFORE sanitize
    weights = model.backbone.sanitize(weights)           # drops model.visual*, remaps names
    params = {f"backbone.{k}": v for k, v in weights.items()}
    params["score.weight"] = score_w
    return params


class SemIfCrossEncoder:
    """Load once, then .predict(pairs) -> (n, 3) probabilities in LABELS order."""

    def __init__(self, path, bits: int | None = None, bs: int = 8, max_len: int = 4096,
                 sort_by_length: bool = True):
        path = resolve(path)
        self.path = path
        self.config = json.loads((path / "config.json").read_text())
        self.template = self.config["nli_template"]   # "Premise: {premise}\nHypothesis: {hypothesis}"
        self.pad_id = self.config["pad_token_id"]
        self.bs, self.max_len, self.sort_by_length = bs, max_len, sort_by_length

        self.model = SemIfMLX(self.config)
        self.model.update(tree_unflatten(list(load_params(self.model, path).items())))
        mx.eval(self.model.parameters())                 # load full precision FIRST
        if bits is not None:
            # group_size=64: blocks of 64 weights share one scale. Backbone only --
            # the 3x2560 head stays full precision, where rounding is all downside.
            nn.quantize(self.model.backbone, group_size=64, bits=bits)
            mx.eval(self.model.parameters())
        self.model.eval()

        self.tok = AutoTokenizer.from_pretrained(str(path))
        # predict() pads by hand, so this only matters if you later let the tokenizer
        # pad -- and then it matters a lot: the head pools the LAST non-pad token, so
        # padding must go after it, never before.
        self.tok.padding_side = "right"

    def encode(self, premise: str, hypothesis: str) -> list[int]:
        """Token ids for one pair. Truncates the PREMISE, never the packed string.

        The template puts the hypothesis last, so tail truncation would silently
        delete it and leave the head pooling a cut-off premise.
        """
        premise, hypothesis = premise.strip(), hypothesis.strip()
        ids = self.tok(self.template.format(premise=premise, hypothesis=hypothesis))["input_ids"]
        if len(ids) <= self.max_len:
            return ids
        hyp_only = self.template.format(premise="", hypothesis=hypothesis)
        budget = self.max_len - len(self.tok(hyp_only)["input_ids"])
        if budget < 1:
            raise ValueError("hypothesis alone exceeds max_len")
        kept = self.tok.decode(self.tok(premise, add_special_tokens=False)["input_ids"][:budget])
        return self.tok(self.template.format(premise=kept, hypothesis=hypothesis)
                        )["input_ids"][:self.max_len]

    def predict(self, pairs) -> np.ndarray:
        seqs = [self.encode(p, h) for p, h in pairs]
        # A batch costs as much as its longest member: sorting by length before
        # batching cut padded tokens roughly in half on MNLI. Order is restored below.
        order = np.argsort([len(s) for s in seqs], kind="stable") if self.sort_by_length \
            else np.arange(len(seqs))
        out = np.zeros((len(seqs), 3), dtype=np.float32)
        for i in range(0, len(seqs), self.bs):
            idx = order[i:i + self.bs]
            chunk = [seqs[j] for j in idx]
            width = max(len(s) for s in chunk)
            ids = np.full((len(chunk), width), self.pad_id, dtype=np.int32)
            for row, s in enumerate(chunk):
                ids[row, :len(s)] = s          # right-padded: every layer is causal
            logits = self.model(mx.array(ids), mx.array([len(s) for s in chunk]))
            out[idx] = np.array(mx.softmax(logits.astype(mx.float32), axis=-1))
        return out

    def entail(self, premise: str, hypotheses: list[str]) -> np.ndarray:
        """P(entailment) of each hypothesis against one premise."""
        return self.predict([(premise, h) for h in hypotheses])[:, ENT]


if __name__ == "__main__":
    import sys
    ce = SemIfCrossEncoder(sys.argv[1] if len(sys.argv) > 1 else "openjev/qwen3.5-4b-nli-v5")
    probs = ce.predict([("A man is playing a guitar.", "Someone is making music.")])
    print({k: round(float(v), 4) for k, v in zip(LABELS, probs[0])})
