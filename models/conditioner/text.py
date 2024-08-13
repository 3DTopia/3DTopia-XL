import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip
from dva.io import load_from_config

class TextConditioner(nn.Module):
    def __init__(
        self,
        encoder_config,
    ):
        super().__init__()
        self.encoder = load_from_config(encoder_config)

    @torch.no_grad()
    def forward(self, batch, rm, amp=False, precision_dtype=torch.float32):
        assert 'caption_token' in batch, "No tokenized caption in current batch for text conditions"
        caption_token = batch['caption_token']
        with torch.autocast(device_type='cuda', dtype=precision_dtype, enabled=amp):
            results = self.encoder(caption_token)
        return results

class CLIPTextEncoder(nn.Module):
    def __init__(
        self,
        pretrained_path: str,
        model_spec: str = 'ViT-L-14',
    ):
        super().__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(model_spec, pretrained=pretrained_path)
        self.model.eval()

    @torch.no_grad()
    def forward(self, text):
        text_features = self.model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features[:, None, :]
