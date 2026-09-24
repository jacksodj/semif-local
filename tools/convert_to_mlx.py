#!/usr/bin/env python
"""Convert an openjev cross-encoder checkpoint into a ready-to-load MLX checkpoint.

The output holds the Qwen3.5 text backbone already sanitized for mlx-lm (vision tower
dropped, names remapped, norms shifted, conv1d transposed) plus the 3-class head as
`score.weight`, in bf16. semif_mlx.SemIfCrossEncoder loads it with no conversion step.

    .venv/bin/python tools/convert_to_mlx.py openjev/qwen3.5-4b-nli-v5 dist/semif-4b-nli-v5-mlx

Quantization is applied at load time instead (SemIfCrossEncoder(..., bits=8)), so there is
one checkpoint to publish and precision stays the caller's choice.
"""
import argparse
import json
import pathlib
import shutil
import sys

import mlx.core as mx
from mlx.utils import tree_flatten

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from semif_mlx import SemIfCrossEncoder  # noqa: E402

COPY = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "panel_manifest.json",
        "train_result.json"]
SHARD_BYTES = 5 * 1024 ** 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args()
    src, dst = pathlib.Path(args.src), pathlib.Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    ce = SemIfCrossEncoder(src)
    params = dict(tree_flatten(ce.model.parameters()))

    shards, cur, size = [], {}, 0
    for k in sorted(params):
        v = params[k]
        if cur and size + v.nbytes > SHARD_BYTES:
            shards.append(cur)
            cur, size = {}, 0
        cur[k] = v
        size += v.nbytes
    shards.append(cur)
    index = {"metadata": {"total_size": sum(v.nbytes for v in params.values())}, "weight_map": {}}
    for i, shard in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        mx.save_safetensors(str(dst / name), shard, metadata={"format": "mlx"})
        index["weight_map"].update({k: name for k in shard})
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    config = json.loads((src / "config.json").read_text())
    config.pop("vision_config", None)          # text-only: the vision tower is not shipped
    config["semif_mlx"] = {"format": "sanitized", "head": "score.weight",
                           "labels": ["contradiction", "entailment", "neutral"]}
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    for f in COPY:
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    print(f"wrote {len(shards)} shard(s), {index['metadata']['total_size'] / 1e9:.2f} GB -> {dst}")


if __name__ == "__main__":
    main()
