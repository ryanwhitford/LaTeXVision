from pathlib import Path

import pytest
import torch
from torch import nn

from src.data.latex_tokenizer import VOCAB
from src.models.classifier import SymbolClassifier
from src.models.im2latex import Im2LatexModel, PositionalEncoding2D, load_classifier_trunk


def _untrained_trunk() -> nn.Sequential:
    # Test-only: shape/behavior checks don't need trained weights. The
    # production builder (load_classifier_trunk) refuses to run without them.
    return SymbolClassifier(num_classes=25).features[:-1]


def _small_model() -> Im2LatexModel:
    return Im2LatexModel(_untrained_trunk(), d_model=64, num_layers=1, nhead=4, dim_feedforward=128)


def test_2d_positional_encoding_separates_rows_and_columns():
    d = 64
    pe = PositionalEncoding2D(d)(torch.zeros(1, d, 5, 7))[0]  # (d, H, W)
    row_half, col_half = pe[: d // 2], pe[d // 2 :]
    # Row half depends only on the row; column half only on the column --
    # that independence is the point of 2D (vs flattened 1D) encoding.
    assert torch.allclose(row_half[:, :, 0:1].expand_as(row_half), row_half)
    assert torch.allclose(col_half[:, 0:1, :].expand_as(col_half), col_half)
    assert not torch.allclose(row_half[:, 0], row_half[:, 1])


def test_2d_positional_encoding_requires_d_model_divisible_by_4():
    with pytest.raises(ValueError):
        PositionalEncoding2D(250)


def test_warm_start_refuses_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_classifier_trunk(tmp_path)


def test_trunk_keeps_a_spatial_grid():
    # SymbolClassifier's global pool lives in `features`; dropping only the
    # head would collapse the grid to 1x1.
    out = _untrained_trunk()(torch.zeros(1, 1, 64, 256))
    assert out.shape == (1, 128, 16, 64)


def test_forward_shapes_and_padding_mask():
    model = _small_model()
    images = torch.zeros(2, 1, 32, 64)
    mask = torch.ones(2, 32, 64, dtype=torch.bool)
    mask[1, :, 32:] = False  # second image is only half as wide
    memory, memory_pad = model.encode(images, mask)
    assert memory.shape == (2, 8 * 16, 64)
    assert not memory_pad[0].any()
    assert memory_pad[1].reshape(8, 16)[:, 8:].all() and not memory_pad[1].reshape(8, 16)[:, :8].any()
    logits = model(images, mask, torch.zeros(2, 5, dtype=torch.long))
    assert logits.shape == (2, 5, len(VOCAB))


def test_trunk_batchnorm_stays_frozen_in_train_mode():
    model = _small_model().train()
    bns = [m for m in model.trunk.modules() if isinstance(m, nn.BatchNorm2d)]
    assert bns and all(not bn.training for bn in bns)
    assert model.decoder.training


def test_greedy_decode_returns_one_sequence_per_image():
    model = _small_model().eval()
    out = model.greedy_decode(torch.zeros(3, 1, 32, 64), torch.ones(3, 32, 64, dtype=torch.bool), max_len=6)
    assert len(out) == 3 and all(len(s) <= 5 for s in out)


def test_history_from_log_rebuilds_epochs_for_legacy_resume(tmp_path):
    from src.training.train_im2latex import history_from_log

    log = tmp_path / "train.log"
    log.write_text(
        "2026-09-27 22:44:20,564 INFO epoch 1  train_loss=2.2962  val_loss=1.6447  val_exact=0.0510  (1300s)\n"
        "2026-09-27 22:45:00,000 INFO   step 200/2599  loss=0.8737  (54s)\n"
        "2026-09-27 22:56:52,628 INFO epoch 2  train_loss=1.7119  val_loss=1.0570  val_exact=0.1473  (752s)\n"
    )
    history = history_from_log(log)
    assert [r["epoch"] for r in history] == [1, 2]
    assert history[1]["val_exact_match"] == 0.1473 and history[0]["epoch_seconds"] == 1300
    assert history_from_log(tmp_path / "missing.log") == []



def test_cross_attention_gives_one_distribution_per_token_and_restores_the_decoder():
    model = _small_model().eval()
    images, mask = torch.randn(1, 1, 32, 96), torch.ones(1, 32, 96, dtype=torch.bool)
    before = model.greedy_decode(images, mask, max_len=6)
    token_ids = [VOCAB.index("x"), VOCAB.index("^"), VOCAB.index("{")]
    attn = model.cross_attention(images, mask, token_ids)
    assert attn.shape == (3, 8, 24)  # (tokens, H/4, W/4)
    assert torch.allclose(attn.sum(dim=(1, 2)), torch.ones(3), atol=1e-5)
    # The temporary need_weights patch must not leak into normal decoding.
    assert all("forward_with_weights" not in repr(l.multihead_attn.forward) for l in model.decoder.layers)
    assert model.greedy_decode(images, mask, max_len=6) == before

_RUN_DIR = Path(__file__).resolve().parent.parent / "models/im2latex_v1"
_MANIFEST = Path(__file__).resolve().parent.parent / "data/processed/expressions/manifest.jsonl"


@pytest.mark.skipif(
    not ((_RUN_DIR / "model.pt").exists() and _MANIFEST.exists()),
    reason="needs the trained checkpoint (models/im2latex_v1) and the built expression dataset",
)
def test_trained_checkpoint_accuracy_floor_on_human_test_set():
    # Regression guard for the shipped model: exact match on the 165 held-out
    # human-written test expressions was 50.9% at release. Greedy decoding on
    # CPU is deterministic, so a drop below the floor means preprocessing,
    # tokenization or the checkpoint changed -- not noise.
    from PIL import Image

    from src.data.expression_dataset import load_manifest
    from src.data.latex_tokenizer import canonicalize
    from src.recognition.transformer_engine import TransformerRecognizer

    recognizer = TransformerRecognizer.load(_RUN_DIR, device=torch.device("cpu"))
    records = [r for r in load_manifest(_MANIFEST, ["test"]) if r["source"] != "hasy_synth"]
    assert len(records) == 165
    correct = 0
    for rec in records:
        result = recognizer.recognize(Image.open(_MANIFEST.parent / rec["image"]).convert("L"))
        correct += result is not None and canonicalize(result.tokens) == rec["tokens"]
    assert correct / len(records) >= 0.45
