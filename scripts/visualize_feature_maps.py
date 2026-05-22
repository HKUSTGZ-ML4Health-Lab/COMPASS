import os
import sys
import argparse
from pathlib import Path
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from train_baseline_3d import ResNet383D
from train_compass import SemanticNavigatedModel

# ==========================================
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 150

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ==========================================
# ==========================================
# --- Example patient settings. Override them from CLI for real experiments. ---
P1_CT_PATH = ""
P1_OMVP_PATH = ""
P1_SEQ_ID = 30
P1_TEXT = "Imaging of the right native hip joint reveals blurred trabeculae and heterogeneous density with geographic changes. The articular surface shows no significant collapse with fair joint alignment. An ellipsoid necrotic zone measuring 44.4 x 38.5 x 36.0 mm is identified. The study concludes with right femoral head avascular necrosis."

P2_CT_PATH = ""
P2_OMVP_PATH = ""
P2_SEQ_ID = 56
P2_TEXT_UNPAIRED = "Bilateral native hip joints demonstrate blurred trabeculae, heterogeneous sclerosis, cystic changes, and peripheral osteosclerosis. No significant joint space narrowing or effusion is noted, with preserved articular alignment. A flattened ellipsoid necrotic zone measuring 38.5 x 41.1 x 30.4 mm is identified. The study concludes with Stage II bilateral avascular necrosis of the femoral head."
P2_TEXT_PAIRED = "Post-treatment follow-up of bilateral native hip joints reveals disordered trabeculae, geographic sclerotic bands, and subchondral lucencies with left lateral pillar involvement. Articular surfaces remain smooth with preserved joint spaces. An ellipsoid necrotic zone measuring 43.2 x 41.0 x 36.17 mm is identified. The study concludes with Stage II bilateral avascular necrosis of the femoral heads and new subchondral fracture lines."


# ==========================================
# ==========================================
def generate_feature_map(feature_tensor, target_shape=(64, 64), is_3d=False):
    feat_map, _ = torch.max(feature_tensor, dim=0)
    feat_map = feat_map.squeeze().cpu().detach().numpy()
    feat_map = np.maximum(feat_map, 0)
    if np.max(feat_map) > 0:
        feat_map /= np.max(feat_map)
    feat_resized = cv2.resize(feat_map, target_shape, interpolation=cv2.INTER_CUBIC)

    if is_3d:
        feat_resized = cv2.GaussianBlur(feat_resized, (7, 7), 0)
    else:
        feat_resized = np.power(feat_resized, 0.8)

    feat_resized = (feat_resized - feat_resized.min()) / (feat_resized.max() - feat_resized.min() + 1e-8)
    feat_uint8 = np.uint8(255 * feat_resized)
    feat_color = cv2.applyColorMap(feat_uint8, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(feat_color, cv2.COLOR_BGR2RGB)


# ==========================================
# ==========================================
def generate_heatmap(gray_img_3c, feature_tensor, alpha=0.5, is_3d=False):
    heatmap = torch.mean(feature_tensor, dim=0).squeeze().cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    H, W = gray_img_3c.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_CUBIC)

    if is_3d:
        heatmap_resized = cv2.GaussianBlur(heatmap_resized, (15, 15), 0)
    else:
        heatmap_resized = heatmap_resized ** 2
        threshold = np.percentile(heatmap_resized, 60)
        heatmap_resized = np.maximum(heatmap_resized - threshold, 0)

    if np.max(heatmap_resized) > 0:
        heatmap_resized /= np.max(heatmap_resized)

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    superimposed_img = heatmap_color * alpha + gray_img_3c * (1 - alpha)
    return np.clip(superimposed_img, 0, 255).astype(np.uint8)


# ==========================================
# ==========================================
def load_patient_data(ct_path, omvp_path, device):
    ct_3d_np = np.load(ct_path)
    tensor_3d = torch.from_numpy(ct_3d_np).float().unsqueeze(0).unsqueeze(0).to(device)

    omvp_np = np.load(omvp_path)
    indices = np.linspace(0, omvp_np.shape[0] - 1, 64, dtype=int)
    bag_images = []
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    for i in indices:
        img_arr = omvp_np[i]
        img_pil = Image.fromarray((img_arr.transpose(1, 2, 0) * 255).astype(np.uint8)).convert('RGB')
        img_t = TF.resize(img_pil, (64, 64))
        img_t = TF.to_tensor(img_t)
        img_t = TF.normalize(img_t, mean=mean, std=std)
        bag_images.append(img_t)
    tensor_2d = torch.stack(bag_images).unsqueeze(0).to(device)
    return ct_3d_np, tensor_3d, omvp_np, tensor_2d, indices


def get_best_z(ct_3d_np, omvp_np, target_index):
    best_omvp_axial = omvp_np[target_index, 0, :, :]
    min_mse = float('inf')
    best_z = 0
    for z in range(ct_3d_np.shape[0]):
        slice_z = ct_3d_np[z, :, :]
        if slice_z.shape != (64, 64):
            slice_z = cv2.resize(slice_z, (64, 64))
        mse = np.mean((slice_z - best_omvp_axial) ** 2)
        if mse < min_mse:
            min_mse = mse
            best_z = z
    return best_z


# ==========================================
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Generate COMPASS feature-map visualizations.")
    parser.add_argument("--baseline-ckpt", required=True, help="Path to the trained 3D-CNN baseline checkpoint.")
    parser.add_argument("--compass-ckpt", required=True, help="Path to the trained COMPASS checkpoint.")
    parser.add_argument("--text-model-path", default="ncbi/MedCPT-Query-Encoder", help="Hugging Face id or local text encoder path.")
    parser.add_argument("--p1-ct", default=P1_CT_PATH)
    parser.add_argument("--p1-omvp", default=P1_OMVP_PATH)
    parser.add_argument("--p1-seq-id", type=int, default=P1_SEQ_ID)
    parser.add_argument("--p1-text", default=P1_TEXT)
    parser.add_argument("--p2-ct", default=P2_CT_PATH)
    parser.add_argument("--p2-omvp", default=P2_OMVP_PATH)
    parser.add_argument("--p2-seq-id", type=int, default=P2_SEQ_ID)
    parser.add_argument("--p2-text-unpaired", default=P2_TEXT_UNPAIRED)
    parser.add_argument("--p2-text-paired", default=P2_TEXT_PAIRED)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f">>> Device: {device}")

    model_3d = ResNet383D(in_channels=1, num_classes=2).to(device)
    model_3d.load_state_dict(torch.load(args.baseline_ckpt, map_location=device))
    model_3d.eval()

    model_config_2d = {'ct_model': 'resnet18',
                       'text_model': args.text_model_path,
                       'embed_dim': 256, 'target_size': [64, 64, 64]}
    model_2d = SemanticNavigatedModel(model_config_2d).to(device)
    model_2d.load_state_dict(torch.load(args.compass_ckpt, map_location=device))
    model_2d.eval()

    hooks_data = {}

    def get_hook(name):
        def hook(module, input, output):
            hooks_data[name] = output.clone() if isinstance(output, torch.Tensor) else output

        return hook

    model_3d.layer1.register_forward_hook(get_hook('3d_layer1'))  # Pat 1
    model_3d.layer2.register_forward_hook(get_hook('3d_layer2'))  # Pat 2
    model_2d.backbone.ct_encoder[0].register_forward_hook(get_hook('2d_0'))  # Pat 1
    model_2d.backbone.ct_encoder[-2].register_forward_hook(get_hook('2d_-2'))  # Pat 2 Paired
    model_2d.backbone.ct_encoder[-4].register_forward_hook(get_hook('2d_-4'))  # Pat 2 Unpaired

    print("\n>>> Processing patient/example 1...")
    ct_np_1, tsr_3d_1, omvp_np_1, tsr_2d_1, idx_1 = load_patient_data(args.p1_ct, args.p1_omvp, device)

    with torch.no_grad():
        _ = model_3d(tsr_3d_1)
        tok_1 = model_2d.backbone._tokenize([args.p1_text])
        _ = model_2d.forward_val(tsr_2d_1, tok_1['input_ids'].to(device), tok_1['attention_mask'].to(device))

        feat_3d_p1 = hooks_data['3d_layer1']
        feat_2d_p1 = hooks_data['2d_0'][args.p1_seq_id]

    best_z_1 = get_best_z(ct_np_1, omvp_np_1, idx_1[args.p1_seq_id])

    gray_1 = ct_np_1[best_z_1, :, :]
    gray_1 = ((gray_1 - gray_1.min()) / (gray_1.max() - gray_1.min() + 1e-8) * 255).astype(np.uint8)
    gray_3c_1 = cv2.cvtColor(gray_1, cv2.COLOR_GRAY2RGB)

    idx_3d_1 = min(int((best_z_1 / ct_np_1.shape[0]) * feat_3d_p1.shape[2]), feat_3d_p1.shape[2] - 1)
    feat_3d_slice_1 = feat_3d_p1[0, :, idx_3d_1, :, :]

    img_p1_gray = gray_3c_1
    img_p1_3d = generate_feature_map(feat_3d_slice_1, is_3d=True)
    img_p1_2d = generate_feature_map(feat_2d_p1, is_3d=False)
    p1_images = [img_p1_gray, img_p1_3d, img_p1_2d]

    print(">>> Processing patient/example 2...")
    ct_np_2, tsr_3d_2, omvp_np_2, tsr_2d_2, idx_2 = load_patient_data(args.p2_ct, args.p2_omvp, device)

    with torch.no_grad():
        _ = model_3d(tsr_3d_2)
        feat_3d_p2 = hooks_data['3d_layer2']

        tok_2_p = model_2d.backbone._tokenize([args.p2_text_paired])
        _ = model_2d.forward_val(tsr_2d_2, tok_2_p['input_ids'].to(device), tok_2_p['attention_mask'].to(device))
        feat_2d_p2_paired = hooks_data['2d_-2'][args.p2_seq_id]

        tok_2_up = model_2d.backbone._tokenize([args.p2_text_unpaired])
        _ = model_2d.forward_val(tsr_2d_2, tok_2_up['input_ids'].to(device), tok_2_up['attention_mask'].to(device))
        feat_2d_p2_unpaired = hooks_data['2d_-4'][args.p2_seq_id]

    best_z_2 = get_best_z(ct_np_2, omvp_np_2, idx_2[args.p2_seq_id])

    gray_2 = ct_np_2[best_z_2, :, :]
    gray_2 = ((gray_2 - gray_2.min()) / (gray_2.max() - gray_2.min() + 1e-8) * 255).astype(np.uint8)
    gray_3c_2 = cv2.cvtColor(gray_2, cv2.COLOR_GRAY2RGB)

    idx_3d_2 = min(int((best_z_2 / ct_np_2.shape[0]) * feat_3d_p2.shape[2]), feat_3d_p2.shape[2] - 1)
    feat_3d_slice_2 = feat_3d_p2[0, :, idx_3d_2, :, :]

    img_p2_gray = gray_3c_2
    img_p2_3d = generate_heatmap(gray_3c_2, feat_3d_slice_2, alpha=0.55, is_3d=True)
    img_p2_2d_paired = generate_heatmap(gray_3c_2, feat_2d_p2_paired, alpha=0.55, is_3d=False)
    img_p2_2d_unpaired = generate_heatmap(gray_3c_2, feat_2d_p2_unpaired, alpha=0.55, is_3d=False)
    p2_images = [img_p2_gray, img_p2_3d, img_p2_2d_paired, img_p2_2d_unpaired]

    print("\n>>> Composing the figure layout...")
    fig_w, fig_h = 19.0, 8.2
    fig = plt.figure(figsize=(fig_w, fig_h))

    # ==========================================
    # ==========================================
    img_size = 3.3
    pad = 0.65

    top_w = 3 * img_size + 2 * pad
    top_left = (fig_w - top_w) / 2.0

    cbar_w = img_size * 0.05
    cbar_p = img_size * 0.05
    bot_w = 4 * img_size + 3 * pad + cbar_p + cbar_w
    bot_left = (fig_w - bot_w) / 2.0

    bot_bottom = 0.3
    row_gap = 0.55
    top_bottom = bot_bottom + img_size + row_gap

    rect_top = [top_left/fig_w, top_bottom/fig_h, top_w/fig_w, img_size/fig_h]
    rect_bot = [bot_left/fig_w, bot_bottom/fig_h, bot_w/fig_w, img_size/fig_h]

    grid_top = ImageGrid(fig, rect_top,
                         nrows_ncols=(1, 3),
                         axes_pad=pad,
                         share_all=True)

    titles_top = ['Original CT Slice', '3D-CNN Features', 'COMPASS Features']
    letters_top = ['(a)', '(b)', '(c)']

    for ax, img, title, letter in zip(grid_top, p1_images, titles_top, letters_top):
        ax.imshow(img, interpolation='antialiased')
        ax.set_title(title, fontsize=23, pad=12, fontweight='bold')
        ax.text(0.04, 0.96, letter, transform=ax.transAxes, fontsize=34, color='red', fontweight='bold', va='top', ha='left')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

    grid_bot = ImageGrid(fig, rect_bot,
                         nrows_ncols=(1, 4),
                         axes_pad=pad,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="5%",
                         cbar_pad="5%")

    titles_bot = ['Original CT Slice', '3D-CNN', 'COMPASS (Paired Text)', 'COMPASS (Unpaired Text)']
    letters_bot = ['(d)', '(e)', '(f)', '(g)']

    for ax, img, title, letter in zip(grid_bot, p2_images, titles_bot, letters_bot):
        ax.imshow(img, interpolation='antialiased')
        ax.set_title(title, fontsize=24, pad=12, fontweight='bold')
        ax.text(0.04, 0.96, letter, transform=ax.transAxes, fontsize=34, color='red', fontweight='bold', va='top', ha='left')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

    cmap = plt.get_cmap('jet')
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=grid_bot.cbar_axes[0])
    cbar.ax.tick_params(labelsize=14)
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(1.5)

    output_dir = os.path.join(os.getcwd(), "Paper_Figures_PDF")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "TopTier_Combined_Vis_1.pdf")

    plt.savefig(save_path, format='pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)
    print(f"\nSaved the aligned PDF figure to: {save_path}")
    plt.show()


if __name__ == '__main__':
    main()
