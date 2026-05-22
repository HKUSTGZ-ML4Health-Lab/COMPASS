import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# =========================================================================
class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.zeros(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 * torch.exp(-self.params[i]) * loss + 0.5 * self.params[i]
        return loss_sum, self.params[0], self.params[1]


# =========================================================================
# =========================================================================
class FiLMLayer(nn.Module):
    def __init__(self, text_dim, feature_channels):
        super(FiLMLayer, self).__init__()
        self.fc = nn.Linear(text_dim, 2 * feature_channels)

    def forward(self, x, text_emb):
        # x: [B, C, H, W]
        # text_emb: [B, text_dim]

        if len(text_emb.shape) == 3:
            text_emb = text_emb.squeeze(1)

        params = self.fc(text_emb)  # [B, 2*C]
        gamma, beta = torch.split(params, x.size(1), dim=1)

        # [B, C, 1, 1]
        gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1)
        beta = beta.view(beta.size(0), beta.size(1), 1, 1)

        return (1 + gamma) * x + beta


# =========================================================================
# =========================================================================
class TextModulatedDecoder(nn.Module):
    def __init__(self, in_channels=512, text_dim=256, out_size=64):
        super().__init__()

        # Input: ResNet Layer4 Output.

        self.up0 = nn.ConvTranspose2d(in_channels, 256, kernel_size=2, stride=2)
        self.film0 = FiLMLayer(text_dim, 256)
        self.conv0 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )

        # Block 1: 4x4 -> 8x8
        self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.film1 = FiLMLayer(text_dim, 256)
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )

        # Block 2: 8x8 -> 16x16
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.film2 = FiLMLayer(text_dim, 128)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )

        # Block 3: 16x16 -> 32x32
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.film3 = FiLMLayer(text_dim, 64)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )

        # Block 4: 32x32 -> 64x64
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.film4 = FiLMLayer(text_dim, 32)
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )

        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x, text_emb):

        # 0. 2 -> 4
        x = self.up0(x)
        x = self.film0(x, text_emb)
        x = F.relu(x)
        x = self.conv0(x)

        # 1. 4 -> 8
        x = self.up1(x)
        x = self.film1(x, text_emb)
        x = F.relu(x)
        x = self.conv1(x)

        # 2. 8 -> 16
        x = self.up2(x)
        x = self.film2(x, text_emb)
        x = F.relu(x)
        x = self.conv2(x)

        # 3. 16 -> 32
        x = self.up3(x)
        x = self.film3(x, text_emb)
        x = F.relu(x)
        x = self.conv3(x)

        # 4. 32 -> 64
        x = self.up4(x)
        x = self.film4(x, text_emb)
        x = F.relu(x)
        x = self.conv4(x)

        logits = self.final_conv(x)
        return torch.sigmoid(logits)

