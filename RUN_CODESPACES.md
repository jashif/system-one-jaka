# Run in your existing GitHub Codespace

Keep the old checkpoint as a backup. Do not overwrite it until you've confirmed the new model works.

```bash
cd /workspaces/system-one-jaka
source .venv/bin/activate
python make_data.py
python train.py --epochs 8 --batch-size 4 --lr 0.0003 --freeze-encoder --output checkpoints/system1-v2
python evaluate.py --help
python cli.py --checkpoint checkpoints/system1-v2 --context "The application crashes when the user logs in" --question "Which team should handle this?" --options Billing Sales Technical Account
```

The `--freeze-encoder` mode trains only the scoring head, to reduce CPU cost. It is a baseline, not guaranteed to generalize. The data is synthetic and templated. Inspect the epoch validation accuracy and test accuracy; if results are weak, do not treat the model as production-ready. For more capable learning, unfreeze the encoder with a smaller learning rate, e.g. `--lr 2e-5`, which is much slower on CPU.

If the model remains near uniform, send the epoch metrics; do not keep increasing epochs blindly.
