from pathlib import Path
import argparse
import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score, confusion_matrix, precision_score, \
    recall_score
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from torch import nn
import torch
import torch.nn.functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from compass.datasets import Ortho_OMVP_Manager
from compass.models import OrthoMIL_CLIP
from compass.segmentation import AutomaticWeightedLoss
import wandb

# =========================================================================
# =========================================================================
parser = argparse.ArgumentParser(description='Ortho Semantic Navigated MIL Classification + Mixed Sup Seg')
parser.add_argument('--dataset', default='Ortho_OMVP', type=str, help='dataset name')
parser.add_argument('--csv_path',
                    required=True,
                    type=str, help='Path to the metadata CSV file')
parser.add_argument('--gpu', default=None, type=str, help='CUDA_VISIBLE_DEVICES value, e.g. "0"')
parser.add_argument('--workers', default=4, type=int, help='number of data loader workers')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs')
parser.add_argument('--batch-size', default=64, type=int, help='Batch size in patients')
parser.add_argument('--num-samples', default=64, type=int, help='MIL Bag Size (N)')
parser.add_argument('--learning-rate', default=1e-4, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=1e-5, type=float, help='weight decay')

parser.add_argument('--top_k', default=5, type=int, help='Top-K OMVP slices for segmentation')

parser.add_argument('--pretrain_path',
                    default='',
                    type=str, help='Path to the Stage I pretrained checkpoint')

parser.add_argument('--checkpoint-dir', default='./checkpoint_semantic_mil_seg/', type=Path,
                    help='path to checkpoint directory')
parser.add_argument('--backbone', default='resnet18', type=str, help='backbone name')
parser.add_argument('--text_model_path',
                    default='ncbi/MedCPT-Query-Encoder', type=str,
                    help='Hugging Face model id or local BERT/MedCPT path')
parser.add_argument('--name', default='OMVP_Cla_Seg_top_5_dropout', type=str, help='exp name')


# =========================================================================
# =========================================================================
def dice_coeff(pred, target, smooth=1.):
    """
    Compute the Dice coefficient for multi-channel probability maps.
    """
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()


def dice_loss(pred, target, smooth=1.):
    return 1 - dice_coeff(pred, target, smooth)


def sparse_loss(pred):
    return pred.mean()


def iou_score(pred, target, smooth=1.):
    """
    Compute intersection-over-union after thresholding.
    """
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


# =========================================================================
# =========================================================================
def load_checkpoint(model, checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')

        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            if not name.startswith('backbone.') and not name.startswith('classifier.') and not name.startswith('awl'):
                name = 'backbone.' + name
            new_state_dict[name] = v

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items() if
                           k in model_dict and v.shape == model_dict[k].shape}

        msg = model.load_state_dict(pretrained_dict, strict=False)
        print(f"Loaded keys: {len(pretrained_dict)} | Missing keys: {len(msg.missing_keys)}")

        if any('ct_encoder' in k for k in new_state_dict): print("Visual encoder weights loaded.")
        if any('vision_proj' in k for k in new_state_dict): print("Vision projection weights loaded.")
        if any('text_proj' in k for k in new_state_dict): print("Text projection weights loaded.")

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
    return model


# =========================================================================
# =========================================================================
class SemanticNavigatedModel(nn.Module):
    def __init__(self, config, num_classes=1):
        super().__init__()
        self.backbone = OrthoMIL_CLIP(config)

        if 'resnet50' in config['ct_model'] or 'resnet101' in config['ct_model']:
            self.feat_dim = 2048
        else:
            self.feat_dim = 512

        self.dropout = nn.Dropout(p=0.3)

        self.classifier = nn.Linear(self.feat_dim, num_classes)

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.ct_encoder.parameters():
            param.requires_grad = True

        if hasattr(self.backbone, 'decoder'):
            for param in self.backbone.decoder.parameters():
                param.requires_grad = True

        for param in self.classifier.parameters():
            param.requires_grad = True

        print(">>> Trainable: [ResNet, Classifier, Decoder]")

    def get_counterfactual_text(self, text_emb, targets):
        """
        Build counterfactual text embeddings by pairing each sample with an
        opposite-label report from the same mini-batch when available.
        """
        B = text_emb.shape[0]
        neg_text_emb_list = []

        for i in range(B):
            current_label = targets[i].item()
            opp_indices = (targets != current_label).nonzero(as_tuple=True)[0]

            if len(opp_indices) > 0:
                idx = opp_indices[torch.randint(len(opp_indices), (1,)).item()]
                neg_text_emb_list.append(text_emb[idx])
            else:
                fallback_idx = (i + 1) % B
                neg_text_emb_list.append(text_emb[fallback_idx])

        return torch.stack(neg_text_emb_list)

    def forward_train(self, images, input_ids, attention_mask, targets, top_k=5):
        """
        Training forward pass with classification and grounding outputs.
        """
        B, N, C, H, W = images.shape

        # 1. Backbone
        out = self.backbone(images, input_ids, attention_mask)

        features_flat = out['features_flat']
        features_map = out['features_map']
        instance_emb = out['instance_embeds']
        text_emb = out['text_embeds']

        shuffle_idx = torch.randperm(B * N).to(images.device)
        feat_shuffled = features_flat[shuffle_idx]
        targets = targets.float()
        targets_expanded = targets.view(B, 1, 1).expand(B, N, 1).reshape(B * N, 1)
        target_shuffled = targets_expanded[shuffle_idx]

        feat_shuffled = self.dropout(feat_shuffled)

        logits_cls = self.classifier(feat_shuffled)

        img_emb_grouped = instance_emb.view(B, N, -1)
        text_emb_expanded = text_emb.unsqueeze(1)
        similarity = torch.sum(img_emb_grouped * text_emb_expanded, dim=-1)  # (B, N)

        _, topk_indices = torch.topk(similarity, top_k, dim=1)

        offsets = (torch.arange(B, device=images.device) * N).unsqueeze(1)
        flat_topk_indices = (topk_indices + offsets).view(-1)
        selected_feats = features_map[flat_topk_indices]  # (B*K, C, H', W')

        # --- Case 1: Positive (Match) ---
        pos_text_emb = text_emb.unsqueeze(1).expand(B, top_k, -1).reshape(B * top_k, -1)
        pred_masks_pos = self.backbone.decoder(selected_feats, pos_text_emb)

        # --- Case 2: Negative (Counterfactual) ---
        neg_text_emb_raw = self.get_counterfactual_text(text_emb, targets)
        neg_text_emb = neg_text_emb_raw.unsqueeze(1).expand(B, top_k, -1).reshape(B * top_k, -1)
        pred_masks_neg = self.backbone.decoder(selected_feats, neg_text_emb)

        return {
            'logits_cls': logits_cls,
            'target_cls': target_shuffled,
            'pred_mask_pos': pred_masks_pos,
            'pred_mask_neg': pred_masks_neg,
            'flat_indices': flat_topk_indices
        }

    def forward_val(self, images, input_ids, attention_mask):
        """
        Validation/inference pass with semantic top-k aggregation.
        """
        B, N, C, H, W = images.shape
        out = self.backbone(images, input_ids, attention_mask)

        img_feat_raw = out['features_flat'].view(B, N, -1)

        instance_emb = out['instance_embeds'].view(B, N, -1)
        text_emb = out['text_embeds'].unsqueeze(1)

        similarity = torch.sum(instance_emb * text_emb, dim=-1)
        scale = out.get('logit_scale', torch.tensor(1.0)).exp()
        attn_weights = torch.softmax(similarity * scale, dim=1).unsqueeze(-1)

        patient_feat = torch.sum(img_feat_raw * attn_weights, dim=1)
        logits = self.classifier(patient_feat)

        return logits, attn_weights, out['features_map'], instance_emb, out['text_embeds']


def main():
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.empty_cache()

    print(f'Task: {args.name} | Backbone: {args.backbone} | TopK: {args.top_k}')
    wandb.init(project="compass", name=f"{args.name}_{args.backbone}", config=args)

    manager = Ortho_OMVP_Manager(csv_path=args.csv_path, num_samples=args.num_samples)
    train_loader = torch.utils.data.DataLoader(manager.get_dataset('train'), batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(manager.get_dataset('val'), batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    model_config = {
        'ct_model': args.backbone,
        'text_model': args.text_model_path,
        'embed_dim': 256,
        'target_size': [64, 64, 64]
    }

    model = SemanticNavigatedModel(model_config)
    if args.pretrain_path and os.path.exists(args.pretrain_path):
        model = load_checkpoint(model, args.pretrain_path)
    elif args.pretrain_path:
        print(f"Warning: pretrain_path does not exist: {args.pretrain_path}")
    model = model.cuda()

    # AWL + Optimizer
    awl = AutomaticWeightedLoss(2).cuda()
    # ====================================================
    # ====================================================
    base_lr = args.learning_rate

    backbone_lr = 5e-5

    awl_lr = 0.01

    print(f">>> Differential Learning Rate Strategy:")
    print(f"    1. Backbone (Encoder):      {backbone_lr:.1e} (Slow & Safe)")
    print(f"    2. Heads (Cls/Dec):         {base_lr:.1e} (Standard)")
    print(f"    3. AWL Params:              {awl_lr:.1e} (Fast)")

    backbone_params = [p for p in model.backbone.ct_encoder.parameters() if p.requires_grad]

    head_params = []
    # Classifier
    head_params.extend([p for p in model.classifier.parameters() if p.requires_grad])
    # Decoder
    if hasattr(model.backbone, 'decoder'):
        head_params.extend([p for p in model.backbone.decoder.parameters() if p.requires_grad])

    awl_params = list(awl.parameters())

    optimizer_groups = [
        {'params': backbone_params, 'lr': backbone_lr},  # index 0
        {'params': head_params, 'lr': base_lr},  # index 1
        {'params': awl_params, 'lr': awl_lr}  # index 2
    ]

    optimizer = torch.optim.Adam(
        optimizer_groups,
        weight_decay=args.weight_decay
    )
    # ====================================================

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

    pos_weight = torch.tensor([3.15]).cuda()
    criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = GradScaler()

    best_val_auc = 0.0

    for epoch in tqdm(range(args.epochs)):
        model.train()
        epoch_loss = 0
        dice_sum = 0

        for step, batch_data in enumerate(train_loader):
            images = batch_data['image'].cuda()
            masks = batch_data['mask'].cuda()
            has_mask = batch_data.get('has_mask', torch.ones(images.size(0), 1)).cuda().bool().view(-1)
            targets = batch_data['target'].cuda()
            text_list = batch_data['text']

            token_out = model.backbone._tokenize(text_list)
            input_ids = token_out['input_ids'].cuda()
            attn_mask = token_out['attention_mask'].cuda()

            optimizer.zero_grad()

            with autocast():
                out_dict = model.forward_train(images, input_ids, attn_mask, targets, top_k=args.top_k)

                # 1. Classification Loss

                loss_cls = criterion_cls(out_dict['logits_cls'], out_dict['target_cls'])

                # 2. Segmentation Loss
                B, N, C_mask, H, W = masks.shape
                masks_flat = masks.view(B * N, C_mask, H, W)

                target_mask_pos = masks_flat[out_dict['flat_indices']]
                topk_has_mask = has_mask.view(B, 1).expand(B, args.top_k).reshape(-1)
                if topk_has_mask.any():
                    loss_dice = dice_loss(out_dict['pred_mask_pos'][topk_has_mask], target_mask_pos[topk_has_mask])
                else:
                    loss_dice = torch.tensor(0.0, device=images.device)
                if (~topk_has_mask).any():
                    loss_sparse = sparse_loss(out_dict['pred_mask_pos'][~topk_has_mask])
                else:
                    loss_sparse = torch.tensor(0.0, device=images.device)
                loss_seg_pos = loss_dice + loss_sparse

                target_mask_neg = torch.zeros_like(target_mask_pos)
                loss_seg_neg = F.mse_loss(out_dict['pred_mask_neg'], target_mask_neg)

                loss_seg_total = loss_seg_pos + loss_seg_neg

                # 3. AWL
                loss_final, p1, p2 = awl(loss_cls, loss_seg_total)
                # loss_final = 0.8 * loss_cls + 0.2 * loss_seg_total

            scaler.scale(loss_final).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss_final.item()
            dice_sum += (1 - loss_seg_pos.item())

        scheduler.step()

        # Validation
        is_last_epoch = (epoch == args.epochs - 1)
        val_metrics = infer(model, val_loader, best_val_auc, args.checkpoint_dir, args.name, is_last_epoch, args.top_k)

        current_auc = val_metrics['auc']
        if current_auc > best_val_auc:
            best_val_auc = current_auc
            torch.save(model.state_dict(), args.checkpoint_dir / f"{args.name}_best_model.pth")

        wandb.log({
            'epoch': epoch,
            'train_loss': epoch_loss / len(train_loader),
            'loss_final': loss_final.item(),
            'train_dice_pos': dice_sum / len(train_loader),
            'loss_cls_raw': loss_cls.item(),
            'loss_seg_pos': loss_seg_pos.item(),
            'loss_seg_neg': loss_seg_neg.item(),
            'awl_cls_w': 1.0 / (torch.exp(p1).item() + 1e-5),
            'awl_seg_w': 1.0 / (torch.exp(p2).item() + 1e-5),

            'backbone_lr': optimizer.param_groups[0]['lr'],
            'head_lr': optimizer.param_groups[1]['lr'],
            'awl_lr': optimizer.param_groups[2]['lr'],

            'val_auc': val_metrics['auc'],
            'val_f1': val_metrics['f1'],
            'val_acc': val_metrics['acc'],
            'val_precision': val_metrics['precision'],
            'val_sensitivity': val_metrics['recall'],
            'val_specificity': val_metrics['specificity'],
            'val_best_threshold': val_metrics['threshold'],
            'val_seg_dice': val_metrics['seg_dice'],
            'val_seg_iou': val_metrics['seg_iou']
        })

    wandb.finish()


@torch.no_grad()
def infer(model, loader, best_auc_so_far, checkpoint_dir, exp_name, is_last_epoch, top_k):
    model.eval()

    y_true = []
    y_pred = []
    patient_ids = []

    seg_dice_list = []
    seg_iou_list = []

    for batch_data in loader:
        images = batch_data['image'].cuda()
        targets = batch_data['target'].float()
        masks = batch_data['mask'].cuda()
        has_mask = batch_data.get('has_mask', torch.ones(images.size(0), 1)).cuda().bool().view(-1)
        text_list = batch_data['text']
        paths = batch_data['image_path']

        token_out = model.backbone._tokenize(text_list)
        input_ids = token_out['input_ids'].cuda()
        attn_mask = token_out['attention_mask'].cuda()

        logits, _, features_map, instance_emb, text_emb = model.forward_val(images, input_ids, attn_mask)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

        y_pred.extend(probs)
        y_true.extend(targets.numpy().flatten())
        patient_ids.extend(paths)

        B, N, C_m, H, W = masks.shape

        img_emb_grouped = instance_emb.view(B, N, -1)
        text_emb_expanded = text_emb.unsqueeze(1)
        similarity = torch.sum(img_emb_grouped * text_emb_expanded, dim=-1)
        _, topk_indices = torch.topk(similarity, top_k, dim=1)

        offsets = (torch.arange(B, device=images.device) * N).unsqueeze(1)
        flat_topk_indices = (topk_indices + offsets).view(-1)

        selected_feats = features_map[flat_topk_indices]
        pos_text_emb = text_emb.unsqueeze(1).expand(B, top_k, -1).reshape(B * top_k, -1)

        pred_masks = model.backbone.decoder(selected_feats, pos_text_emb)

        masks_flat = masks.view(B * N, C_m, H, W)
        target_masks = masks_flat[flat_topk_indices]

        topk_has_mask = has_mask.view(B, 1).expand(B, top_k).reshape(-1)
        if topk_has_mask.any():
            seg_dice_list.append(dice_coeff(pred_masks[topk_has_mask], target_masks[topk_has_mask]).item())
            seg_iou_list.append(iou_score(pred_masks[topk_has_mask], target_masks[topk_has_mask]).item())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5

    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_pred)
    numerator = 2 * recall_curve * precision_curve
    denom = recall_curve + precision_curve
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))

    best_idx = np.argmax(f1_scores) if len(f1_scores) > 0 else 0
    best_thresh = thresholds[best_idx] if len(thresholds) > 0 else 0.5

    y_hard = (y_pred > best_thresh).astype(int)

    acc = accuracy_score(y_true, y_hard)
    prec = precision_score(y_true, y_hard, zero_division=0)
    sens = recall_score(y_true, y_hard, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_hard, labels=[0, 1]).ravel()
    spec = tn / (tn + fp + 1e-8)

    max_f1 = f1_scores[best_idx] if len(f1_scores) > 0 else 0

    if (auc > best_auc_so_far or is_last_epoch) and checkpoint_dir:
        df = pd.DataFrame({'patient_id': patient_ids, 'label_gt': y_true, 'prob_avg': y_pred, 'pred_label': y_hard})
        suffix = "best" if auc > best_auc_so_far else "last"
        df.to_csv(checkpoint_dir / f"{exp_name}_patient_level_{suffix}.csv", index=False)
        print(f"--- Patient-level Metrics (AUC: {auc:.4f}) ---")

    return {
        'auc': auc,
        'f1': max_f1 * 100,
        'acc': acc * 100,
        'precision': prec * 100,
        'recall': sens * 100,
        'specificity': spec * 100,
        'threshold': best_thresh,
        'seg_dice': np.mean(seg_dice_list) if seg_dice_list else 0.0,
        'seg_iou': np.mean(seg_iou_list) if seg_iou_list else 0.0
    }


if __name__ == '__main__':
    main()
