import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from transformers import BertModel, BertTokenizer
import torchvision.models as models

from compass.resnet3d import ResNet18, ResNet34, ResNet50, ResNet101
from compass.vit3d import vit_base_3d, vit_small_3d, vit_tiny_3d
from compass.segmentation import TextModulatedDecoder


class MILAttention(nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features):
        att_weights = self.attention_layer(features)
        att_weights = torch.softmax(att_weights, dim=1)
        weighted_features = torch.sum(features * att_weights, dim=1)
        return weighted_features, att_weights


class OrthoMIL_CLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ct_model_name = config['ct_model']
        self.text_model_name = config['text_model']
        self.embed_dim = config.get('embed_dim', 768)

        _target_size = config.get('target_size', (64, 64, 64))
        if len(_target_size) == 3:
            self.slice_size = (_target_size[1], _target_size[2])
        else:
            self.slice_size = _target_size

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # -----------------------------------------------------------
        # -----------------------------------------------------------
        self.use_resnet = False
        if 'resnet' in self.ct_model_name:
            self.use_resnet = True
            print(f"Loading Pretrained {self.ct_model_name} from torchvision...")

            if '18' in self.ct_model_name:
                backbone = models.resnet18(pretrained=True)
                feature_dim = 512
            elif '50' in self.ct_model_name:
                backbone = models.resnet50(pretrained=True)
                feature_dim = 2048
            elif '101' in self.ct_model_name:
                backbone = models.resnet101(pretrained=True)
                feature_dim = 2048
            else:
                backbone = models.resnet34(pretrained=True)
                feature_dim = 512

            self.ct_encoder = nn.Sequential(*list(backbone.children())[:-2])
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        elif 'vit' in self.ct_model_name:
            if 'base' in self.ct_model_name:
                self.ct_encoder = vit_base_3d(in_channels=3, image_size=self.slice_size)
                feature_dim = 768
            elif 'tiny' in self.ct_model_name:
                self.ct_encoder = vit_tiny_3d(in_channels=3, image_size=self.slice_size)
                feature_dim = 192
            else:
                self.ct_encoder = vit_small_3d(in_channels=3, image_size=self.slice_size)
                feature_dim = 384
        else:
            raise ValueError(f"Unknown visual model: {self.ct_model_name}")

        # -----------------------------------------------------------
        # 2. MIL Attention & Projections
        # -----------------------------------------------------------
        self.mil_attention = MILAttention(dim=feature_dim)

        self.vision_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(feature_dim, self.embed_dim)
        )

        # -----------------------------------------------------------
        # -----------------------------------------------------------
        self.lm_model = BertModel.from_pretrained(self.text_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.text_model_name)

        self.text_proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(768, self.embed_dim)
        )

        # -----------------------------------------------------------
        # -----------------------------------------------------------
        self.decoder = TextModulatedDecoder(
            in_channels=feature_dim,
            text_dim=self.embed_dim,
            out_size=64
        )

    def _tokenize(self, text):
        if isinstance(text, str):
            text = [text]
        input_ids = []
        attention_masks = []
        for sent in text:
            encoded_dict = self.tokenizer.encode_plus(
                sent, add_special_tokens=True, max_length=256,
                truncation_strategy='longest_first', return_tensors='pt'
            )
            curr_ids = encoded_dict['input_ids']
            if curr_ids.shape[1] < 256:
                padding = torch.zeros((1, 256 - curr_ids.shape[1]), dtype=torch.long)
                curr_ids = torch.cat([curr_ids, padding], dim=1)
            curr_mask = (curr_ids != 0).long()
            input_ids.append(curr_ids)
            attention_masks.append(curr_mask)
        return {'input_ids': torch.cat(input_ids, dim=0), 'attention_mask': torch.cat(attention_masks, dim=0)}

    def get_text_emb(self, input_ids, attention_mask):
        outputs = self.lm_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs[1]

    def forward(self, omvp_images, input_ids, attention_mask):
        """
        Return patient-level embeddings, instance embeddings, and spatial maps
        used by the semantic top-k grounding decoder.
        """
        B, N, C, H, W = omvp_images.shape
        images_flat = omvp_images.view(B * N, C, H, W)

        features_map = self.ct_encoder(images_flat)

        if self.use_resnet:
            if len(features_map.shape) == 4:
                features_flat = self.global_pool(features_map).flatten(1)
            else:
                features_flat = features_map
        else:
            features_flat = features_map

        features_seq = features_flat.view(B, N, -1)
        patient_embedding_raw, attn_weights = self.mil_attention(features_seq)

        patient_emb = self.vision_proj(patient_embedding_raw)
        patient_emb = normalize(patient_emb, dim=-1)

        text_feat_raw = self.get_text_emb(input_ids, attention_mask)
        text_emb = self.text_proj(text_feat_raw)
        text_emb = normalize(text_emb, dim=-1)

        instance_emb = self.vision_proj(features_flat)  # (B*N, Embed_Dim)
        instance_emb = normalize(instance_emb, dim=-1)

        return {
            'image_embeds': patient_emb,
            'text_embeds': text_emb,
            'instance_embeds': instance_emb,
            'features_map': features_map,
            'features_flat': features_flat,
            'attn_weights': attn_weights,
            'logit_scale': self.logit_scale
        }
