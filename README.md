---
title: rizzo-pii — live demo
emoji: 🦔
colorFrom: purple
colorTo: indigo
sdk: static
pinned: false
license: mit
models:
  - rizzoaiacademy/rizzo-pii-0.3B
  - r69shabh/rizzo-pii-0.3B-onnx-web
short_description: Reversible PII anonymization for Italian legal text
---

# rizzo-pii — live demo

Try [rizzo-pii](https://github.com/Rizzo-AI-Academy/rizzo-pii) in the browser: paste an Italian contract,
ruling or email, see the 22 PII types highlighted (codice fiscale, P.IVA, IBAN, dati catastali…), copy the
placeholder text into any LLM, then restore the real values from its answer.

Runs 100% client-side: the model (int8 ONNX of `rizzoaiacademy/rizzo-pii-0.3B`) runs in onnxruntime-web,
and the project's own regex/checksum net (`detectors.py`, unmodified) plus the merge/placeholder/restore
logic (ported from `src/app/app.py`) run in Pyodide. Nothing you paste is uploaded.

Code and model © Rizzo AI Academy, MIT License (`LICENSE-rizzo-pii`). Unofficial community demo.

---
Live demo: https://huggingface.co/spaces/r69shabh/rizzo-pii-demo
