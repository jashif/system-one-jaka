# System 1 V3 — shared-context transformer decision model

Experimental Laya-inspired architecture, **not Laya**. Encodes the context once per request, each question once, and each option once, then jointly compares options with a 2-layer transformer decision head. Unlike V2, supports multiple questions in a single call. Encoder representations are still computed for each distinct question and option; this is not a single ModernBERT pass for the whole request.

## GitHub Codespaces (4 CPU cores / 16 GB RAM)

Use a **new folder**; do not overwrite your V2 checkpoint. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python train.py --epochs 8 --batch-size 2 --lr 0.0003 --freeze-encoder --output checkpoints/system1-v3
python evaluate.py --checkpoint checkpoints/system1-v3 --data data/test.jsonl
python calibrate.py --checkpoint checkpoints/system1-v3 --data data/val.jsonl
python evaluate.py --checkpoint checkpoints/system1-v3 --data data/test.jsonl
python cli.py --checkpoint checkpoints/system1-v3 --context 'The application crashes when the user logs in' --question 'Which team should handle this?' --options Billing Sales Technical Account
uvicorn api:app --host 0.0.0.0 --port 8000
```

The first training run downloads ModernBERT-base weights. CPU training may take a long time. Training from scratch in V3 is necessary: V2's head checkpoint is not compatible with V3's transformer head. Training saves the best validation-loss checkpoint, not necessarily the highest-accuracy checkpoint. The `--freeze-encoder` setting trains only the decision head.

API docs: `http://localhost:8000/docs` (forward port 8000 in Codespaces).

POST `/decide`:

```json
{"context":"Customer was charged twice", "questions":[{"question":"Department?", "options":["Billing","Sales","Technical"]},{"question":"Priority?", "options":["Low","Medium","High"]}]}
```

## Data

The bundled `data/*.jsonl` files are V2's small synthetic customer-support examples. Each line can be either a legacy single-question record (`context`, `question`, `options`, `label`) or a multi-question record (`context`, `questions`: array of `{question, options, label}`). Labels are zero-based option indices. The same question can have any number of options >=2. **These data are not a serious generalization benchmark**; the training/test templates may overlap. Add real and template-disjoint datasets before drawing conclusions.

## Metrics and caveats

`evaluate.py` reports accuracy, log loss, multiclass Brier score, 10-bin ECE and per-request latency. `calibrate.py` fits a positive softmax temperature **only on validation data**, writing `temperature.json`; `evaluate.py` then applies it on test data. ECE on a 32-example test set is unstable. Compare V2 and V3 on identical examples and hardware. V3 may be slower due to multiple encoder calls and attention overhead; no speed or accuracy gains are guaranteed. V3 does not implement RLCD or Laya's exact architecture.
