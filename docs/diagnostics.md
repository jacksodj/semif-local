# Diagnostic order

If your numbers disappoint, check these in order. They're sorted by how cheap each check is against how
likely it is to pay off, which is roughly the opposite of where instinct sends you.

1. **The verbalizer.** Print the exact hypothesis strings. A statement *about the input* ("This message is
   about the X project") is a much weaker question than a claim *about its content* ("The email discusses X").
   `sweep_templates.py` does this sweep.
2. **The name actually appears in the text.** Score a known-correct example against the canonical label and
   against whatever the text calls it. If the canonical version scores near zero, you have a vocabulary
   problem. The fix is aliases that phrase the hypothesis, chosen by the prefilter, so each project still
   costs one model call.
3. **What else is in the question.** Strip descriptions, glosses and parentheticals down to the bare name.
   Each extra clause is one more thing the model has to find support for. Descriptions belong in the prefilter.
4. **The premise contains the signal, and the hypothesis survived.** Decode the token ids you actually sent.
   If you truncate the packed string, the hypothesis (which comes last) is the first thing lost, and nothing
   tells you. `SemIfCrossEncoder.encode` truncates the premise only.
5. **Numeric precision** (the dtype, not the metric). Re-run one benchmark at bf16 before believing any
   quantized number: `eval_mlx.py --compare-quant`.
6. **The right answer is in the option list.** Measure prefilter recall separately from ranking accuracy
   (`benchmark_capability.py` prints recall by K).
7. **Labels are trustworthy and classes have enough examples.** Historical tags from a busy person aren't
   a gold standard.
8. **Only then, a bigger model.**

Two deployment hazards to monitor, whatever the accuracy:

- **The catch-all trap.** When a project is missing from the taxonomy, its mail can be auto-tagged
  confidently into a generic bucket that nobody reviews. Audit that bucket, and treat growth in its volume as
  a sign the label list has drifted.
- **Prompt injection.** The model isn't hardened against it: one adversarial line in the premise can flip a
  decision. Keep deterministic checks in front of anything security-relevant.
