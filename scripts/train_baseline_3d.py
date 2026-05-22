from pathlib import Path
import argparse
import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, \
    precision_recall_curve, confusion_matrix
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from compass.datasets_cla import Ortho_CT_TEXT_Dataset_Manager

# =========================================================================
# =========================================================================
parser = argparse.ArgumentParser(description='Ortho CT 3D ResNet38 Finetuning')
parser.add_argument('--dataset', default='Ortho_CT', type=str, help='dataset name')
parser.add_argument('--csv_path',
                    required=True,
                    type=str, help='Path to the metadata CSV file')
parser.add_argument('--gpu', default=None, type=str, help='CUDA_VISIBLE_DEVICES value, e.g. "0"')
parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loader workers')
parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=4, type=int, metavar='N', help='Batch Size')
parser.add_argument('--learning-rate', default=5e-4, type=float, metavar='LR', help='learning rate')
parser.add_argument('--checkpoint-dir', default='./checkpoint_finetune/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--in_channels', default=1, type=int, help='input channels')
parser.add_argument('--name', default='Ortho_ResNet38', type=str, metavar='B', help='exp name')


# =========================================================================
# =========================================================================
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=1, downsample=None):
        super().__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet383D(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.inplanes = 64
        self.expansion = 4

        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 3, stride=2)
        self.layer3 = self._make_layer(256, 4, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512 * self.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * self.expansion),
            )
        layers = []
        layers.append(ResidualBlock3D(self.inplanes, planes, stride, self.expansion, downsample))
        self.inplanes = planes * self.expansion
        for _ in range(1, blocks):
            layers.append(ResidualBlock3D(self.inplanes, planes, expansion=self.expansion))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, None


def main():
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)
    torch.cuda.empty_cache()

    print(f'Task: {args.name} | Model: ResNet383D')
    wandb.init(project="compass-baselines", name=f"{args.name}_3D_CNN", config=args)

    manager = Ortho_CT_TEXT_Dataset_Manager(csv_path=args.csv_path)
    train_dataset = manager.get_dataset('train')
    val_dataset = manager.get_dataset('val')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    print(">>> Loading Model (ResNet383D matched to resnet38.py)...")
    model_train = ResNet383D(in_channels=args.in_channels, num_classes=2).to('cuda')

    # =========================================================================
    # =========================================================================
    model_lr = args.learning_rate
    optimizer = optim.Adam(model_train.parameters(), lr=model_lr)

    criterion_cls = nn.CrossEntropyLoss()
    scaler = GradScaler()

    def adjust_learning_rate(optimizer, epoch):
        model_lrnew = model_lr * (0.1 ** (epoch // 10))
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Adjusting LR to {model_lrnew}")
        for param_group in optimizer.param_groups:
            param_group["lr"] = model_lrnew

    best_val_auc = 0.0

    for epoch in tqdm(range(args.epochs)):
        adjust_learning_rate(optimizer, epoch)

        model_train.train()
        epoch_loss = 0

        for step, batch_data in enumerate(train_loader):
            img = batch_data['ct'].to('cuda')
            target = batch_data['target'].to('cuda')

            optimizer.zero_grad()

            current_target = target.view(-1).long()

            with autocast():
                logits_cls, _ = model_train(img)  # logits_cls shape: [B, 2]
                loss = criterion_cls(logits_cls, current_target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        is_last_epoch = (epoch == args.epochs - 1)
        val_results = infer(
            model_train,
            val_loader,
            best_auc_so_far=best_val_auc,
            checkpoint_dir=args.checkpoint_dir,
            exp_name=args.name,
            is_last_epoch=is_last_epoch
        )

        current_auc = val_results['linear']['auc']
        if current_auc > best_val_auc:
            best_val_auc = current_auc
            torch.save(model_train.state_dict(), args.checkpoint_dir / f"{args.name}_best_model.pth")

        m = val_results['linear']
        wandb.log({
            'epoch': epoch,
            'train_loss': epoch_loss / len(train_loader),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'val_auc': m['auc'],
            'val_acc': m['acc'],
            'val_f1': m['f1'],
            'val_precision': m['precision'],
            'val_sensitivity': m['recall'],
            'val_specificity': m['specificity'],
            'val_best_threshold': m['threshold']
        })

    wandb.finish()


@torch.no_grad()
def infer(model_train, loader, best_auc_so_far=-1.0, checkpoint_dir=None, exp_name="exp", is_last_epoch=False):
    model_train.eval()

    y_true_list = []
    y_pred_list = []
    patient_ids = []

    for batch_data in loader:
        img = batch_data['ct'].to('cuda')
        target = batch_data['target']
        paths = batch_data['path']

        logits_linear, _ = model_train(img)  # [B, 2]

        probs_linear = torch.softmax(logits_linear, dim=1)[:, 1].cpu().numpy().flatten()

        y_true_list.extend(target.cpu().numpy().flatten())
        y_pred_list.extend(probs_linear)
        patient_ids.extend(paths)

    y_true_patient = np.array(y_true_list)
    y_pred_patient = np.array(y_pred_list)

    def compute_metrics(y_true, y_pred):
        auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_pred)
        numerator = 2 * recall_curve * precision_curve
        denom = recall_curve + precision_curve
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))

        best_idx = np.argmax(f1_scores) if len(f1_scores) > 0 else 0
        best_thresh = thresholds[best_idx] if len(thresholds) > 0 else 0.5
        max_f1 = f1_scores[best_idx] if len(f1_scores) > 0 else 0

        y_hard = (y_pred > best_thresh).astype(int)
        acc = accuracy_score(y_true, y_hard)
        prec = precision_score(y_true, y_hard, zero_division=0)
        sens = recall_score(y_true, y_hard, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_true, y_hard, labels=[0, 1]).ravel()
        spec = tn / (tn + fp + 1e-8)

        return {
            'auc': auc, 'acc': acc * 100, 'f1': max_f1 * 100,
            'precision': prec * 100, 'recall': sens * 100, 'specificity': spec * 100,
            'threshold': best_thresh, 'y_hard': y_hard
        }

    metrics_patient = compute_metrics(y_true_patient, y_pred_patient)

    if (metrics_patient['auc'] > best_auc_so_far or is_last_epoch) and checkpoint_dir:
        df_res = pd.DataFrame({
            'patient_id': patient_ids,
            'label_gt': y_true_patient,
            'prob_avg': y_pred_patient,
            'pred_label': metrics_patient['y_hard'],
            'success': (metrics_patient['y_hard'] == y_true_patient).astype(int)
        })
        suffix = "best" if metrics_patient['auc'] > best_auc_so_far else "last"
        df_res.to_csv(checkpoint_dir / f"{exp_name}_patient_level_{suffix}.csv", index=False)
        print(f"--- Patient-level Metrics (AUC: {metrics_patient['auc']:.4f}) ---")

    return {'linear': metrics_patient}


if __name__ == '__main__':
    main()
