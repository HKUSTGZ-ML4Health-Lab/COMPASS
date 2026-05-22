"""Stage I semantic-space alignment for OMVP bags and clinical reports."""

from pathlib import Path
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from compass.datasets import Ortho_OMVP_Manager
from compass.models import OrthoMIL_CLIP


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clip_loss(image_embeds, text_embeds, logit_scale):
    logits_per_image = logit_scale.exp() * image_embeds @ text_embeds.t()
    logits_per_text = logits_per_image.t()
    labels = torch.arange(image_embeds.shape[0], device=image_embeds.device)
    return (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2


def main():
    parser = argparse.ArgumentParser(description="Train the Stage I OMVP-text alignment model.")
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--text_model_path", default="ncbi/MedCPT-Query-Encoder")
    parser.add_argument("--checkpoint-dir", default="./checkpoints/stage1", type=Path)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-samples", default=64, type=int)
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-5, type=float)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    manager = Ortho_OMVP_Manager(args.csv_path, num_samples=args.num_samples)
    train_loader = torch.utils.data.DataLoader(
        manager.get_dataset("train"),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        manager.get_dataset("val"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = OrthoMIL_CLIP(
        {
            "ct_model": args.backbone,
            "text_model": args.text_model_path,
            "embed_dim": 256,
            "target_size": [64, 64, 64],
        }
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler()

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            images = batch["image"].cuda(non_blocking=True)
            tokenized = model._tokenize(batch["text"])
            input_ids = tokenized["input_ids"].cuda(non_blocking=True)
            attention_mask = tokenized["attention_mask"].cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                output = model(images, input_ids, attention_mask)
                loss = clip_loss(output["image_embeds"], output["text_embeds"], output["logit_scale"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].cuda(non_blocking=True)
                tokenized = model._tokenize(batch["text"])
                input_ids = tokenized["input_ids"].cuda(non_blocking=True)
                attention_mask = tokenized["attention_mask"].cuda(non_blocking=True)
                output = model(images, input_ids, attention_mask)
                val_losses.append(clip_loss(output["image_embeds"], output["text_embeds"], output["logit_scale"]).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.checkpoint_dir / "stage1_alignment_best.pth")


if __name__ == "__main__":
    main()
