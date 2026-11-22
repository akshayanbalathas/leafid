"""Train one experiment, resumably.

Examples
  # baseline CNN from scratch on lab + field
  python -m leafid.train --model simple_cnn --epochs 30 --lr 1e-3 --run simple_cnn

  # linear probe: frozen ImageNet backbone, only the new head trains
  python -m leafid.train --model resnet50 --epochs 10 --freeze-epochs 10 --run resnet50_probe

  # two-stage fine-tune: 3 epochs head-only, then unfreeze everything
  python -m leafid.train --model resnet50 --epochs 15 --freeze-epochs 3 --run resnet50_full

  # lab-only training (for the lab -> field domain-shift experiment)
  python -m leafid.train --model resnet50 --train-sources lab --run resnet50_labonly

Surviving Colab disconnects
  Use --save-dir on Google Drive and add --resume. After every epoch the full training
  state (model, optimizer, LR schedule, history) is written to checkpoints/<run>_last.pt.
  Re-running the exact same command with --resume continues from the next epoch.
  The best model (by field validation accuracy) is kept in checkpoints/<run>.pt.

Every run uses the same splits/splits.csv, so results are directly comparable.
"""
import argparse
import csv
import os
import time
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from .config import SEED, checkpoint_dir, reports_dir
from .data import image_root, load_classes, make_loader, split_version
from .evaluate import accuracy, predict_loader
from .models import MODEL_NAMES, build_model, head_parameters, save_checkpoint, set_backbone_trainable

HISTORY_FIELDS = ["epoch", "frozen", "train_loss", "train_acc",
                  "val_lab_top1", "val_field_top1", "val_field_top5", "secs"]
# Changing these between a run and its --resume would silently mix two experiments.
MUST_MATCH = ("model", "train_sources", "epochs", "freeze_epochs", "lr", "backbone_lr_mult",
              "weight_decay", "label_smoothing", "batch_size", "no_augment")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODEL_NAMES, required=True)
    ap.add_argument("--run", required=True, help="name for checkpoint + report folder")
    ap.add_argument("--train-sources", default="lab,field", help="lab | field | lab,field")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--freeze-epochs", type=int, default=0, help="epochs with backbone frozen (head only)")
    ap.add_argument("--lr", type=float, default=1e-3, help="head learning rate")
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1, help="pretrained backbone lr = lr * this")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--save-dir", type=Path, default=None,
                    help="where checkpoints/ and reports/ go (default: $LEAFID_OUTPUT or the repo root)")
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/<run>_last.pt if it exists")
    return ap.parse_args()


def atomic_save(obj, path: Path) -> None:
    """Write to a temp file then rename, so a disconnect mid-save can't corrupt the checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] no GPU found: training will be very slow (Colab: Runtime -> Change runtime type -> T4 GPU)")
    classes = load_classes()
    sources = tuple(s.strip() for s in args.train_sources.split(","))

    ckpt_dir, report_dir = checkpoint_dir(args.save_dir), reports_dir(args.save_dir) / args.run
    best_path, last_path = ckpt_dir / f"{args.run}.pt", ckpt_dir / f"{args.run}_last.pt"
    history_path = report_dir / "history.csv"
    report_dir.mkdir(parents=True, exist_ok=True)
    split_v = split_version()
    print(f"images from: {image_root()}\nsplit:       {split_v}\nsaving to:   {ckpt_dir}  and  {report_dir}")

    train_loader = make_loader("train", sources, train=True, augment=not args.no_augment,
                               batch_size=args.batch_size, num_workers=args.num_workers)
    val_loaders = {s: make_loader("val", (s,), batch_size=args.batch_size * 2, num_workers=args.num_workers)
                   for s in ("lab", "field")}

    model = build_model(args.model, len(classes)).to(device)
    head = head_parameters(model)
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    # From-scratch models train every layer at the full lr; the pretrained one uses a smaller backbone lr.
    bb_lr = args.lr * (args.backbone_lr_mult if args.model == "resnet50" else 1.0)
    optimizer = torch.optim.AdamW([{"params": head, "lr": args.lr}, {"params": backbone, "lr": bb_lr}],
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    start_epoch, best = 1, -1.0
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        changed = [k for k in MUST_MATCH if state["args"].get(k) != vars(args)[k]]
        if state.get("split_version") != split_v:
            raise SystemExit(f"--resume refused: {last_path.name} was trained on split "
                             f"{state.get('split_version', 'v1 (unversioned)')}, splits.csv is now {split_v}. "
                             f"Delete or move the old checkpoint to start this run again.")
        if changed:
            raise SystemExit(f"--resume refused: {changed} differ from the saved run. "
                             f"Use the same command, or a new --run name.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best = state["epoch"] + 1, state["best"]
        print(f"resumed {args.run} after epoch {state['epoch']} (best val field top1 so far = {best:.3f})")
    else:
        if args.resume:
            print(f"--resume: no {last_path.name} yet, starting fresh")
        with history_path.open("w", newline="") as f:
            csv.writer(f).writerow(HISTORY_FIELDS)

    if start_epoch > args.epochs:
        print(f"{args.run} already finished all {args.epochs} epochs; nothing to do")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        frozen = epoch <= args.freeze_epochs
        set_backbone_trainable(model, not frozen)
        model.train()
        t0, loss_sum, correct, seen = time.time(), 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}{' (frozen)' if frozen else ''}"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        scheduler.step()

        val = {s: accuracy(*predict_loader(model, loader, device)) for s, loader in val_loaders.items()}
        row = [epoch, frozen, round(loss_sum / seen, 4), round(correct / seen, 4),
               val["lab"]["top1"], val["field"]["top1"], val["field"]["top5"], round(time.time() - t0, 1)]
        with history_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)
        print(dict(zip(HISTORY_FIELDS, row)))

        # Select on field accuracy: real-world photos are what the app sees.
        score = val["field"]["top1"] or 0.0
        if score > best:
            best = score
            save_checkpoint(best_path, model, args.model, classes,
                            {"args": vars(args), "epoch": epoch, "val": val, "split_version": split_v})
            print(f"  saved best -> {best_path.name} (val field top1 = {best:.3f})")

        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": epoch, "best": best, "args": vars(args), "split_version": split_v}, last_path)

    last_path.unlink(missing_ok=True)  # finished: the resumable state isn't needed any more (~300 MB)
    print(f"done. best val field top1 = {best:.3f}\n  model:   {best_path}\n  history: {history_path}")


if __name__ == "__main__":
    main()
