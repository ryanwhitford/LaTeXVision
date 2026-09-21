"""Investigate a reported bias: is 'div' over-predicted under distribution
shift (as opposed to on clean HASYv2 data, where it's already confirmed to
have near-perfect precision)?

For each robustness condition, records the full predicted-class distribution
(not just accuracy), so we can see whether errors concentrate into one
"sink" class rather than spreading out across plausible confusions.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.api.inference import SymbolPredictor  # noqa: E402
from src.evaluation.robustness import CONDITIONS  # noqa: E402
from src.training.train_classifier import resolve_device  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
run_dir = REPO_ROOT / "models" / "symbol_classifier_v1"

device = resolve_device("auto")
predictor = SymbolPredictor.load(run_dir, device=device)

df = pd.read_csv(run_dir / "test_split.csv")
df = df.sample(n=300, random_state=42).reset_index(drop=True)

rng = np.random.default_rng(42)

print(f"{'condition':<18} {'top predicted classes (with counts)'}")
for condition_name, condition_fn in CONDITIONS.items():
    pred_counter = Counter()
    wrong_pred_counter = Counter()  # only when the prediction was WRONG
    for _, row in df.iterrows():
        image = Image.open(row["path"])
        perturbed = condition_fn(image, rng)
        predictions = predictor.predict(perturbed, top_k=1)
        if predictions is None:
            pred_counter["<no-prediction>"] += 1
            continue
        pred_counter[predictions[0].class_name] += 1
        if predictions[0].class_name != row["class_name"]:
            wrong_pred_counter[predictions[0].class_name] += 1

    top5 = pred_counter.most_common(5)
    top5_wrong = wrong_pred_counter.most_common(5)
    print(f"\n{condition_name}:")
    print(f"  all predictions:   {top5}")
    print(f"  WRONG predictions: {top5_wrong}")
