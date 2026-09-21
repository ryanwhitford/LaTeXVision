"""Regression test for experiments/audit/AUDIT_REPORT.md, Finding 1:
training/eval and the deployed inference path silently used different
preprocessing (crop_to_content applied only at inference), which measured
96% "clean" accuracy while the real served pipeline scored ~40% on
equivalent input. Fixed by making both call get_transforms(content_crop=...)
with the same flag, sourced from class_mapping.json. This test fails loudly
if that ever drifts apart again.
"""

from pathlib import Path

import pytest
import torch
from PIL import Image

from src.api.inference import SymbolPredictor
from src.data.hasy import HASYSymbolDataset, load_dataset_config, load_labels
from src.data.preprocessing import get_transforms
from src.training.train_classifier import resolve_device

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models" / "symbol_classifier_v1"

requires_trained_model = pytest.mark.skipif(
    not (MODEL_DIR / "model.pt").exists(),
    reason="No trained model at models/symbol_classifier_v1",
)


@requires_trained_model
def test_dataset_and_predictor_use_the_same_content_crop_setting():
    import json

    with open(MODEL_DIR / "class_mapping.json") as f:
        class_mapping = json.load(f)

    predictor = SymbolPredictor.load(MODEL_DIR, device=torch.device("cpu"))

    # The predictor's transform must have been built with the same
    # content_crop flag the checkpoint was actually trained with -- not a
    # hardcoded assumption.
    training_content_crop = class_mapping["content_crop"]
    expected_transform = get_transforms(
        class_mapping["image_size"], train=False, content_crop=training_content_crop
    )
    assert repr(predictor.transform) == repr(expected_transform)


@requires_trained_model
def test_training_dataloader_and_predictor_agree_on_the_same_image():
    """The train/eval Dataset path and the SymbolPredictor path must produce
    the same prediction for the same image -- they're the two ways this
    exact image could be shown to the model, and they must match."""
    dataset_config = load_dataset_config(REPO_ROOT / "configs" / "dataset.yaml")
    if not dataset_config.labels_path.exists():
        pytest.skip("HASYv2 raw data not downloaded; see README.md")

    df = load_labels(dataset_config)
    sample = df.iloc[0]

    device = resolve_device("cpu")
    predictor = SymbolPredictor.load(MODEL_DIR, device=device)

    # Path A: what train_classifier.py / evaluate_classifier.py feed the model.
    eval_transform = get_transforms(dataset_config.image_size, train=False, content_crop=True)
    dataset = HASYSymbolDataset(df.iloc[[0]], transform=eval_transform)
    tensor_a, _ = dataset[0]
    with torch.no_grad():
        logits_a = predictor.model(tensor_a.unsqueeze(0).to(device))

    # Path B: what the FastAPI /recognize endpoint feeds the model.
    image = Image.open(sample["path"]).convert("L")
    predictions_b = predictor.predict(image, top_k=1)

    predicted_class_a = predictor.class_names[logits_a.argmax(dim=1).item()]
    assert predicted_class_a == predictions_b[0].class_name
