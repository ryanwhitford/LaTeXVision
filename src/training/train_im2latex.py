"""Train the image-to-LaTeX transformer on the combined expression dataset.

    python -m src.training.train_im2latex
    python -m src.training.train_im2latex --epochs 1 --max-train 500   # smoke test
    python -m src.training.train_im2latex --resume   # continue run_dir's run

Model selection uses validation exact match (greedy decoding), never the
test split -- the test split belongs to src/evaluation/expression_benchmark.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import time
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data.expression_dataset import BucketBatchSampler, ExpressionDataset, collate, load_manifest
from src.data.latex_tokenizer import PAD, TOKEN_TO_ID, decode
from src.models.im2latex import build_from_classifier
from src.training.train_classifier import resolve_device

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/im2latex.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--run-dir")
    parser.add_argument("--max-train", type=int, help="cap training samples (smoke test)")
    parser.add_argument("--max-val", type=int, help="cap validation samples (smoke test)")
    parser.add_argument("--resume", action="store_true", help="continue the run in run_dir instead of starting over")
    return parser.parse_args()


_EPOCH_LINE = re.compile(
    r"epoch (\d+)\s+train_loss=([\d.]+)\s+val_loss=([\d.]+)\s+val_exact=([\d.]+)\s+\((\d+)s\)"
)


def history_from_log(log_path: Path) -> list[dict]:
    """Rebuild per-epoch history from train.log, for runs that predate
    last.pt (their optimizer state was never saved, only model.pt)."""
    rows = {}
    for m in _EPOCH_LINE.finditer(log_path.read_text() if log_path.exists() else ""):
        epoch, train_loss, val_loss, val_exact, secs = m.groups()
        rows[int(epoch)] = {
            "epoch": int(epoch), "train_loss": float(train_loss), "val_loss": float(val_loss),
            "val_exact_match": float(val_exact), "epoch_seconds": float(secs),
        }
    return [rows[e] for e in sorted(rows)]


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> dict:
    model.eval()
    total_loss, n_tokens, exact, n = 0.0, 0, 0, 0
    for b, (images, mask, tgt_in, tgt_out) in enumerate(loader):
        if device.type == "mps" and b % 5 == 0:
            torch.mps.empty_cache()  # greedy decoding visits every prefix length
        images, mask, tgt_in, tgt_out = images.to(device), mask.to(device), tgt_in.to(device), tgt_out.to(device)
        logits = model(images, mask, tgt_in)
        non_pad = (tgt_out != TOKEN_TO_ID[PAD]).sum().item()
        total_loss += criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)).item() * non_pad
        n_tokens += non_pad
        for pred, gold in zip(model.greedy_decode(images, mask), tgt_out.tolist()):
            exact += decode(pred) == decode(gold)
            n += 1
    return {"val_loss": total_loss / max(n_tokens, 1), "val_exact_match": exact / max(n, 1)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.run_dir:
        cfg["output"]["run_dir"] = args.run_dir
    tcfg = cfg["training"]

    random.seed(tcfg["seed"])
    torch.manual_seed(tcfg["seed"])
    device = resolve_device(tcfg["device"])

    root = Path(cfg["data"]["root"])
    manifest = Path(cfg["data"]["manifest"])
    train_records = load_manifest(manifest, ["train"], cfg["data"]["train_sources"])
    val_records = load_manifest(manifest, ["val"], cfg["data"]["val_sources"])
    # <bos> + tokens + <eos> must fit the decoder's positional table. Drop
    # the few over-long TRAINING samples (14 of ~50k, all MathWriting
    # synthetic); never drop evaluation samples -- if one didn't fit, the
    # model would be silently truncated on it, so fail loudly instead.
    max_len = cfg["model"]["max_len"]
    too_long = [r for r in train_records if len(r["tokens"]) + 2 > max_len]
    train_records = [r for r in train_records if len(r["tokens"]) + 2 <= max_len]
    logger.info("dropped %d training samples longer than max_len=%d", len(too_long), max_len)
    eval_max = max(len(r["tokens"]) + 2 for r in load_manifest(manifest, ["val", "test"]))
    if eval_max > max_len:
        raise ValueError(f"an evaluation sample needs {eval_max} positions > max_len={max_len}")
    random.Random(tcfg["seed"]).shuffle(train_records)
    if args.max_train:
        train_records = train_records[: args.max_train]
    if args.max_val:
        val_records = val_records[: args.max_val]
    logger.info("train=%d val=%d device=%s", len(train_records), len(val_records), device)

    bs = cfg["data"]["batch_size"]
    train_ds = ExpressionDataset(train_records, root, augment=cfg["data"]["augment"])
    val_ds = ExpressionDataset(val_records, root)
    logger.info("preloading + normalizing images (shared serving preprocessing)...")
    max_px = cfg["data"]["max_pixels_per_batch"]
    train_loader = DataLoader(
        train_ds,
        batch_sampler=BucketBatchSampler(train_ds.sizes(), bs, max_pixels=max_px, seed=tcfg["seed"]),
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_sampler=BucketBatchSampler(val_ds.sizes(), 64, max_pixels=max_px, seed=0), collate_fn=collate
    )

    model = build_from_classifier(
        Path(cfg["encoder"]["classifier_run"]), freeze_trunk_batchnorm=cfg["encoder"]["freeze_batchnorm"], **cfg["model"]
    ).to(device)
    trunk_params = list(model.trunk.parameters())
    trunk_ids = {id(p) for p in trunk_params}
    other_params = [p for p in model.parameters() if id(p) not in trunk_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": trunk_params, "lr": tcfg["learning_rate"] * cfg["encoder"]["trunk_lr_multiplier"]},
            {"params": other_params, "lr": tcfg["learning_rate"]},
        ],
        weight_decay=tcfg["weight_decay"],
    )
    total_steps = tcfg["epochs"] * len(train_loader)

    def lr_lambda(step: int) -> float:
        if step < tcfg["warmup_steps"]:
            return (step + 1) / tcfg["warmup_steps"]
        progress = (step - tcfg["warmup_steps"]) / max(1, total_steps - tcfg["warmup_steps"])
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=TOKEN_TO_ID[PAD], label_smoothing=tcfg["label_smoothing"])
    eval_criterion = nn.CrossEntropyLoss(ignore_index=TOKEN_TO_ID[PAD])

    run_dir = Path(cfg["output"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model_config.json").write_text(json.dumps({**cfg["model"], "freeze_trunk_batchnorm": cfg["encoder"]["freeze_batchnorm"]}, indent=2))
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    history, best, stale, start_epoch = [], -1.0, 0, 1
    if args.resume:
        last = run_dir / "last.pt"
        if last.exists():
            state = torch.load(last, map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            history, best, stale = state["history"], state["best"], state["stale"]
        else:
            # Legacy run: only the best weights exist. Resume from them with
            # the LR schedule fast-forwarded; AdamW's moment estimates restart
            # from zero (they re-settle within a few hundred steps).
            history = history_from_log(run_dir / "train.log")
            if not history or not (run_dir / "model.pt").exists():
                raise FileNotFoundError(f"nothing to resume in {run_dir}")
            model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
            best_row = max(history, key=lambda r: r["val_exact_match"])
            best, stale = best_row["val_exact_match"], history[-1]["epoch"] - best_row["epoch"]
            if stale:
                logger.warning("model.pt is from epoch %d, not the last logged epoch %d", best_row["epoch"], history[-1]["epoch"])
            done = history[-1]["epoch"] * len(train_loader)
            scheduler.last_epoch = done
            for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                group["lr"] = base_lr * lr_lambda(done)
        start_epoch = history[-1]["epoch"] + 1
        train_loader.batch_sampler.epoch = start_epoch - 1  # keep the per-epoch shuffle sequence
        logger.info("resuming at epoch %d (best val_exact=%.4f, stale=%d, lr=%.2e)",
                    start_epoch, best, stale, optimizer.param_groups[1]["lr"])

    for epoch in range(start_epoch, tcfg["epochs"] + 1):
        model.train()
        start, running, steps = time.time(), 0.0, 0
        for images, mask, tgt_in, tgt_out in train_loader:
            images, mask, tgt_in, tgt_out = images.to(device), mask.to(device), tgt_in.to(device), tgt_out.to(device)
            logits = model(images, mask, tgt_in)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            running += loss.item()
            steps += 1
            if device.type == "mps" and steps % 25 == 0:
                torch.mps.empty_cache()  # see H_BUCKET note in expression_dataset.py
            if steps % 200 == 0:
                logger.info("  step %d/%d  loss=%.4f  (%.0fs)", steps, len(train_loader), running / steps, time.time() - start)
        metrics = evaluate(model, val_loader, device, eval_criterion)
        if device.type == "mps":
            torch.mps.empty_cache()
        row = {"epoch": epoch, "train_loss": running / max(steps, 1), **metrics, "epoch_seconds": time.time() - start}
        history.append(row)
        logger.info(
            "epoch %d  train_loss=%.4f  val_loss=%.4f  val_exact=%.4f  (%.0fs)",
            epoch, row["train_loss"], row["val_loss"], row["val_exact_match"], row["epoch_seconds"],
        )
        if metrics["val_exact_match"] > best:
            best, stale = metrics["val_exact_match"], 0
            torch.save(model.state_dict(), run_dir / "model.pt")
            logger.info("  new best val_exact=%.4f -> saved", best)
        else:
            stale += 1
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "history": history, "best": best, "stale": stale,
        }, run_dir / "last.pt")
        if stale >= tcfg["early_stopping_patience"]:
            logger.info("early stopping")
            break

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "summary.json").write_text(json.dumps({
        "best_val_exact_match": best,
        "epochs_run": len(history),
        "train_size": len(train_records),
        "val_size": len(val_records),
        "param_count": sum(p.numel() for p in model.parameters()),
        "total_training_seconds": sum(r["epoch_seconds"] for r in history),
        "device": str(device),
    }, indent=2))


if __name__ == "__main__":
    main()
