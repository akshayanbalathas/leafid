"""Train one experiment.

Examples
  # baseline CNN from scratch on lab + field
  python -m leafid.train --model simple_cnn --epochs 30 --lr 1e-3 --run simple_cnn

  # linear probe: frozen ImageNet backbone, only the new head trains
  python -m leafid.train --model resnet50 --epochs 10 --freeze-epochs 10 --run resnet50_probe

  # two-stage fine-tune: 3 epochs head-only, then unfreeze everything
  python -m leafid.train --model resnet50 --epochs 15 --freeze-epochs 3 --run resnet50_full

  # lab-only training (for the lab -> field domain-shift experiment)
  python -m leafid.train --model resnet50 --train-sources lab --run resnet50_labonly

Every run uses the same splits/splits.csv, so results are directly comparable.
Validation always reports lab and field accuracy separately.
"""
import argparse
import csv
import time

import torch
from torch import nn
from tqdm import tqdm

from .config import CHECKPOINT_DIR, REPORTS_DIR, SEED
from .data import load_classes, make_loader
from .evaluate import accuracy, predict_loader
from .models import MODEL_NAMES, build_model, head_parameters, save_checkpoint, set_backbone_trainable


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODEL_NAMES, required=True)
    ap.add_argument("--run", required=True, help="name for checkpoint + report folder")
    ap.add_argument("--train-sources", default="lab,field", help="lab | field | lab,field")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--freeze-epochs", type=int, default=0, help="epochs with backbone frozen (head only)")
    ap.add_argument("--lr", type=float, default=1e-3, help="head learning rate")
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1, help="backbone lr = lr * this")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = load_classes()
    sources = tuple(s.strip() for s in args.train_sources.split(","))

    train_loader = make_loader("train", sources, train=True, augment=not args.no_augment,
                               batch_size=args.batch_size, num_workers=args.num_workers)
    val_loaders = {s: make_loader("val", (s,), batch_size=args.batch_size * 2, num_workers=args.num_workers)
                   for s in ("lab", "field")}

    model = build_model(args.model, len(classes)).to(device)
    head = head_parameters(model)
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    # From-scratch models train every layer at the full lr; pretrained ones use a smaller backbone lr.
    bb_lr = args.lr * (args.backbone_lr_mult if args.model == "resnet50" else 1.0)
    optimizer = torch.optim.AdamW([{"params": head, "lr": args.lr}, {"params": backbone, "lr": bb_lr}],
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    report_dir = REPORTS_DIR / args.run
    report_dir.mkdir(parents=True, exist_ok=True)
    history_path = report_dir / "history.csv"
    fields = ["epoch", "frozen", "train_loss", "train_acc", "val_lab_top1", "val_field_top1", "val_field_top5", "secs"]
    with history_path.open("w", newline="") as f:
        csv.writer(f).writerow(fields)

    best = -1.0
    for epoch in range(1, args.epochs + 1):
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
        row = [epoch, frozen, loss_sum / seen, correct / seen,
               val["lab"]["top1"], val["field"]["top1"], val["field"]["top5"], round(time.time() - t0, 1)]
        with history_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)
        print(dict(zip(fields, row)))

        # Select on field accuracy: real-world photos are what the app sees.
        score = val["field"]["top1"] or 0.0
        if score > best:
            best = score
            save_checkpoint(CHECKPOINT_DIR / f"{args.run}.pt", model, args.model, classes,
                            {"args": vars(args), "epoch": epoch, "val": val})
            print(f"  saved best (val field top1 = {best:.3f})")

    print(f"done. best val field top1 = {best:.3f}; history in {history_path}")


if __name__ == "__main__":
    main()
