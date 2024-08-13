# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch as th
import numpy as np

import logging

from .vgg import VGGLossMasked

logger = logging.getLogger("dva.{__name__}")

class DCTLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, inputs, preds, iteration=None):
        loss_dict = {"loss_total": 0.0}
        target = inputs['gt']
        recon = preds['recon']
        posterior = preds['posterior']
        fft_gt = th.view_as_real(th.fft.fft(target.reshape(target.shape[0], -1)))
        fft_recon = th.view_as_real(th.fft.fft(recon.reshape(recon.shape[0], -1)))
        loss_recon_dct_l1 = th.mean(th.abs(fft_gt - fft_recon))
        loss_recon_l1 = th.mean(th.abs(target - recon))
        loss_kl = posterior.kl().mean()
        loss_dict.update(loss_recon_l1=loss_recon_l1, loss_recon_dct_l1=loss_recon_dct_l1, loss_kl=loss_kl)
        loss_total = self.weights.recon * loss_recon_dct_l1 + self.weights.kl * loss_kl

        loss_dict["loss_total"] = loss_total
        return loss_total, loss_dict

class VAESepL2Loss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, inputs, preds, iteration=None):
        loss_dict = {"loss_total": 0.0}
        target = inputs['gt']
        recon = preds['recon']
        posterior = preds['posterior']
        recon_diff = (target - recon) ** 2
        loss_recon_sdf_l1 = th.mean(recon_diff[:, 0:1, ...])
        loss_recon_rgb_l1 = th.mean(recon_diff[:, 1:4, ...])
        loss_recon_mat_l1 = th.mean(recon_diff[:, 4:6, ...])
        loss_kl = posterior.kl().mean()
        loss_dict.update(loss_sdf_l1=loss_recon_sdf_l1, loss_rgb_l1=loss_recon_rgb_l1, loss_mat_l1=loss_recon_mat_l1, loss_kl=loss_kl)
        loss_total = self.weights.sdf * loss_recon_sdf_l1 + self.weights.rgb * loss_recon_rgb_l1 + self.weights.mat * loss_recon_mat_l1
        if "kl" in self.weights:
            loss_total += self.weights.kl * loss_kl

        loss_dict["loss_total"] = loss_total
        return loss_total, loss_dict

class VAESepLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, inputs, preds, iteration=None):
        loss_dict = {"loss_total": 0.0}
        target = inputs['gt']
        recon = preds['recon']
        posterior = preds['posterior']
        recon_diff = th.abs(target - recon)
        loss_recon_sdf_l1 = th.mean(recon_diff[:, 0:1, ...])
        loss_recon_rgb_l1 = th.mean(recon_diff[:, 1:4, ...])
        loss_recon_mat_l1 = th.mean(recon_diff[:, 4:6, ...])
        loss_kl = posterior.kl().mean()
        loss_dict.update(loss_sdf_l1=loss_recon_sdf_l1, loss_rgb_l1=loss_recon_rgb_l1, loss_mat_l1=loss_recon_mat_l1, loss_kl=loss_kl)
        loss_total = self.weights.sdf * loss_recon_sdf_l1 + self.weights.rgb * loss_recon_rgb_l1 + self.weights.mat * loss_recon_mat_l1
        if "kl" in self.weights:
            loss_total += self.weights.kl * loss_kl

        loss_dict["loss_total"] = loss_total
        return loss_total, loss_dict

class VAELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, inputs, preds, iteration=None):
        loss_dict = {"loss_total": 0.0}
        target = inputs['gt']
        recon = preds['recon']
        posterior = preds['posterior']
        loss_recon_l1 = th.mean(th.abs(target - recon))
        loss_kl = posterior.kl().mean()
        loss_dict.update(loss_recon_l1=loss_recon_l1, loss_kl=loss_kl)
        loss_total = self.weights.recon * loss_recon_l1 + self.weights.kl * loss_kl

        loss_dict["loss_total"] = loss_total
        return loss_total, loss_dict

class PrimSDFLoss(nn.Module):
    def __init__(self, weights, shape_opt_steps=2000, tex_opt_steps=6000):
        super().__init__()
        self.weights = weights
        self.shape_opt_steps = shape_opt_steps
        self.tex_opt_steps = tex_opt_steps
    
    def forward(self, inputs, preds, iteration=None):
        loss_dict = {"loss_total": 0.0}

        if iteration < self.shape_opt_steps:
            target_sdf = inputs['sdf']
            sdf = preds['sdf']
            loss_sdf_l1 = th.mean(th.abs(sdf - target_sdf))
            loss_dict.update(loss_sdf_l1=loss_sdf_l1)
            loss_total = self.weights.sdf_l1 * loss_sdf_l1

            prim_scale = preds["prim_scale"]
            # we use 1/scale instead of the original 100/scale as our scale is normalized to [-1, 1] cube
            if "vol_sum" in self.weights:
                loss_prim_vol_sum = th.mean(th.sum(th.prod(1 / prim_scale, dim=-1), dim=-1))
                loss_dict.update(loss_prim_vol_sum=loss_prim_vol_sum)
                loss_total += self.weights.vol_sum * loss_prim_vol_sum
        
        if iteration >= self.shape_opt_steps and iteration < self.tex_opt_steps:
            target_tex = inputs['tex']
            tex = preds['tex']
            loss_tex_l1 = th.mean(th.abs(tex - target_tex))
            loss_dict.update(loss_tex_l1=loss_tex_l1)
            
            loss_total = (
                self.weights.rgb_l1 * loss_tex_l1
            )
            if "mat_l1" in self.weights:
                target_mat = inputs['mat']
                mat = preds['mat']
                loss_mat_l1 = th.mean(th.abs(mat - target_mat))
                loss_dict.update(loss_mat_l1=loss_mat_l1)
                loss_total += self.weights.mat_l1 * loss_mat_l1

        if "grad_l2" in self.weights:
            loss_grad_l2 = th.mean((preds["grad"] - inputs["grad"]) ** 2)
            loss_total += self.weights.grad_l2 * loss_grad_l2
            loss_dict.update(loss_grad_l2=loss_grad_l2)

        loss_dict["loss_total"] = loss_total
        return loss_total, loss_dict


class TotalMVPLoss(nn.Module):
    def __init__(self, weights, assets=None):
        super().__init__()

        self.weights = weights

        if "vgg" in self.weights:
            self.vgg_loss = VGGLossMasked()

    def forward(self, inputs, preds, iteration=None):

        loss_dict = {"loss_total": 0.0}

        B = inputs["image"].shape

        # rgb
        target_rgb = inputs["image"].permute(0, 2, 3, 1)
        # removing the mask
        target_rgb = target_rgb * inputs["image_mask"][:, 0, :, :, np.newaxis]

        rgb = preds["rgb"]
        loss_rgb_mse = th.mean(((rgb - target_rgb) / 16.0) ** 2.0)
        loss_dict.update(loss_rgb_mse=loss_rgb_mse)

        alpha = preds["alpha"]

        # mask loss
        target_mask = inputs["image_mask"][:, 0].to(th.float32)
        loss_mask_mae = th.mean((target_mask - alpha).abs())
        loss_dict.update(loss_mask_mae=loss_mask_mae)

        B = alpha.shape[0]

        # beta prior on opacity
        loss_alpha_prior = th.mean(
            th.log(0.1 + alpha.reshape(B, -1))
            + th.log(0.1 + 1.0 - alpha.reshape(B, -1))
            - -2.20727
        )
        loss_dict.update(loss_alpha_prior=loss_alpha_prior)

        prim_scale = preds["prim_scale"]
        loss_prim_vol_sum = th.mean(th.sum(th.prod(100.0 / prim_scale, dim=-1), dim=-1))
        loss_dict.update(loss_prim_vol_sum=loss_prim_vol_sum)

        loss_total = (
            self.weights.rgb_mse * loss_rgb_mse
            + self.weights.mask_mae * loss_mask_mae
            + self.weights.alpha_prior * loss_alpha_prior
            + self.weights.prim_vol_sum * loss_prim_vol_sum
        )

        if "embs_l2" in self.weights:
            loss_embs_l2 = th.sum(th.norm(preds["embs"], dim=1))
            loss_total += self.weights.embs_l2 * loss_embs_l2
            loss_dict.update(loss_embs_l2=loss_embs_l2)

        if "vgg" in self.weights:
            loss_vgg = self.vgg_loss(
                rgb.permute(0, 3, 1, 2),
                target_rgb.permute(0, 3, 1, 2),
                inputs["image_mask"],
            )
            loss_total += self.weights.vgg * loss_vgg
            loss_dict.update(loss_vgg=loss_vgg)

        if "prim_scale_var" in self.weights:
            log_prim_scale = th.log(prim_scale)
            # NOTE: should we detach this?
            log_prim_scale_mean = th.mean(log_prim_scale, dim=1, keepdim=True)
            loss_prim_scale_var = th.mean((log_prim_scale - log_prim_scale_mean) ** 2.0)
            loss_total += self.weights.prim_scale_var * loss_prim_scale_var
            loss_dict.update(loss_prim_scale_var=loss_prim_scale_var)

        loss_dict["loss_total"] = loss_total

        return loss_total, loss_dict


def process_losses(loss_dict, reduce=True, detach=True):
    """Preprocess the dict of losses outputs."""
    result = {
        k.replace("loss_", ""): v for k, v in loss_dict.items() if k.startswith("loss_")
    }
    if detach:
        result = {k: v.detach() for k, v in result.items()}
    if reduce:
        result = {k: float(v.mean().item()) for k, v in result.items()}
    return result
