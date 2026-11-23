# System 1 Dynamic Decision Model

A trainable **ModernBERT cross-encoder** that scores a variable set of candidate options. Each candidate is encoded jointly with the context and question; scores are normalized across the supplied options. This is **not** a reproduction of Laya and does **not** guarantee calibrated probabilities without adequate data.

## Setup

Python 3.10+ recommended. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first training run downloads ModernBERT-base from Hugging Face (~hundreds of MB). Internet access is needed.

## Data

`data/train.jsonl`, `data/val.jsonl`, and `data/test.jsonl` are **small synthetic demo sets**, not production-quality training data. Each JSONL record has:

```json
{"context":"Charged twice for my subscription", "question":"Which team should handle this?", "options":["Billing", "Sales", "Technical support"], "label":0}
```

`label` is the zero-based index of the correct option. Options can vary in number and order. For serious training, gather many diverse human-labeled examples, ensure train/test separation by source and time, and avoid label leakage.

## Train

```bash
python train.py --epochs 3 --batch-size 4 --output checkpoints/system1
```

On CPU this can be slow. GPU recommended. The script saves the best validation-loss checkpoint, tokenizer, and configuration.

## Evaluate and calibrate

```bash
python evaluate.py --data data/test.jsonl
python calibrate.py --data data/val.jsonl
cat checkpoints/system1/calibration.json
python evaluate.py --data data/test.jsonl --temperature YOUR_FITTED_TEMPERATURE
```

Calibration is optional and must be fitted on held-out validation data. The API accepts an explicit temperature (default 1.0); it does not silently apply calibration.

## Predict from CLI

```bash
python cli.py --context 'Customer was charged twice' --question 'Which team should handle this?' --options Billing Sales Technical
```

## Start API

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

Try:

```bash
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"context":"Customer was charged twice","question":"Which team should handle this?","options":["Billing","Sales","Technical support"]}'
```

Open http://127.0.0.1:8000/docs for Swagger UI. Do not expose this unauthenticated API to the public internet.

## Design limitations

- Candidate options are processed as separate sequences, batched into a single encoder invocation. Runtime and memory grow with the number of options; this is not one constant-cost encoding of the entire decision.
- A fixed maximum sequence length truncates long context and options.
- Softmax values are relative to the options supplied and may be overconfident.
- There is no abstention, open-ended generation, RL training, or out-of-distribution detector.
- Demo examples are intentionally tiny; meaningful accuracy requires a larger, diverse dataset.
- Latency metrics include tokenization and Python overhead, and depend on hardware.

## Tests

```bash
pytest -q
```
