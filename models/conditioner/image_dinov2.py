import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, InterpolationMode, Normalize

import logging
logger = logging.getLogger(__name__)




class Dinov2Wrapper(nn.Module):
    """
    Dino v2 wrapper using original implementation, hacked with modulation.
    """
    def __init__(self, model_name: str, modulation_dim: int = None, freeze: bool = True):
        super().__init__()
        self.modulation_dim = modulation_dim
        self.model = self._build_dinov2(model_name, modulation_dim=modulation_dim)
        self.preprocess = Compose([
            Resize(self.model.patch_embed.img_size[0], interpolation=InterpolationMode.BICUBIC),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        if freeze:
            if modulation_dim is not None:
                raise ValueError("Modulated Dinov2 requires training, freezing is not allowed.")
            self._freeze()

    def _freeze(self):
        logger.warning(f"======== Freezing Dinov2Wrapper ========")
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    @staticmethod
    def _build_dinov2(model_name: str, modulation_dim: int = None, pretrained: bool = True):
        from importlib import import_module
        dinov2_hub = import_module(".dinov2.hub.backbones", package=__package__)
        model_fn = getattr(dinov2_hub, model_name)
        logger.info(f"Modulation dim for Dinov2 is {modulation_dim}.")
        model = model_fn(modulation_dim=modulation_dim, pretrained=pretrained)
        return model

    # @torch.compile
    def forward(self, image: torch.Tensor, mod: torch.Tensor = None):
        # image: [N, H, W, C] -- need to be permuted!!!
        # mod: [N, D] or None
        assert image.shape[-1] == 3
        image = image.permute(0, 3, 1, 2) / 255.
        image = self.preprocess(image)
        if self.modulation_dim is None:
            assert mod is None, "Unexpected modulation input in dinov2 forward."
            outs = self.model(image, is_training=True)
        else:
            assert mod is not None, "Modulation input is required in modulated dinov2 forward."
            outs = self.model(image, mod=mod, is_training=True)
        ret = torch.cat([
            outs["x_norm_clstoken"].unsqueeze(dim=1),
            outs["x_norm_patchtokens"],
        ], dim=1)
        # ret in [B, 1370, 384]
        return ret
