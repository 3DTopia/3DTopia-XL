import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode

import open_clip
from dva.io import load_from_config

def sample_orbit_traj(radius, height, start_theta, end_theta, num_points, world_up=torch.Tensor([0, 1, 0])):
    # return [num_points, 3, 4]
    angles = torch.rand((num_points, )) * (end_theta - start_theta) + start_theta
    return get_pose_on_orbit(radius=radius, height=height, angles=angles, world_up=world_up)

def get_pose_on_orbit(radius, height, angles, world_up=torch.Tensor([0, 1, 0])):
    num_points = angles.shape[0]
    x = radius * torch.cos(angles)
    h = torch.ones((num_points,)) * height
    z = radius * torch.sin(angles)
    position = torch.stack([x, h, z], dim=-1)
    forward = position / torch.norm(position, p=2, dim=-1, keepdim=True)
    right = -torch.cross(world_up[None, ...], forward)
    right /= torch.norm(right, dim=-1, keepdim=True)
    up = torch.cross(forward, right)
    up /= torch.norm(up, p=2, dim=-1, keepdim=True)
    rotation = torch.stack([right, up, forward], dim=1)
    translation = torch.Tensor([0, 0, radius])[None, :, None].repeat(num_points, 1, 1)
    return torch.concat([rotation, translation], dim=2)

class DummyImageConditioner(nn.Module):
    def __init__(
        self,
        num_prims,
        dim_feat,
        prim_shape,
        encoder_config,
        sample_view=False,
        sample_start=torch.pi*0.25,
        sample_end=torch.pi*0.75,
    ):
        super().__init__()

        self.num_prims = num_prims
        self.dim_feat = dim_feat
        self.prim_shape = prim_shape
        self.sample_view = sample_view
        self.sample_start = sample_start
        self.sample_end = sample_end
        self.encoder = None

    @torch.no_grad()
    def forward(self, batch, rm, amp, precision_dtype=torch.float32):
        return batch['cond']

class ImageConditioner(nn.Module):
    def __init__(
        self,
        num_prims,
        dim_feat,
        prim_shape,
        encoder_config,
        sample_view=False,
        sample_start=torch.pi*0.25,
        sample_end=torch.pi*0.75,
    ):
        super().__init__()

        self.num_prims = num_prims
        self.dim_feat = dim_feat
        self.prim_shape = prim_shape
        self.sample_view = sample_view
        self.sample_start = sample_start
        self.sample_end = sample_end
        self.encoder = load_from_config(encoder_config)
    
    def sdf2alpha(self, sdf):
        return torch.exp(-(sdf / 0.005) ** 2)

    @torch.no_grad()
    def forward(self, batch, rm, amp, precision_dtype=torch.float32):
        # TODO: replace with real rendering process in primsdf
        assert 'input_param' in batch, "No parameters in current batch for rendering image conditions"
        prim_volume = batch['input_param']
        bs = prim_volume.shape[0]
        preds = {}
        geo_start_index = 4
        geo_end_index = geo_start_index + self.prim_shape ** 3 # non-inclusive
        tex_start_index = geo_end_index
        tex_end_index = tex_start_index + self.prim_shape ** 3 * 3 # non-inclusive
        feat_geo = prim_volume[:, :, geo_start_index: geo_end_index]
        feat_tex = prim_volume[:, :, tex_start_index: tex_end_index]
        prim_alpha = self.sdf2alpha(feat_geo).reshape(bs, self.num_prims, 1, self.prim_shape, self.prim_shape, self.prim_shape) * 255
        prim_rgb = feat_tex.reshape(bs, self.num_prims, 3, self.prim_shape, self.prim_shape, self.prim_shape) * 255
        preds['prim_rgba'] = torch.concat([prim_rgb, prim_alpha], dim=2)
        pos = prim_volume[:, :, 1:4]
        scale = prim_volume[:, :, 0:1]
        preds['prim_pos'] = pos.reshape(bs, self.num_prims, 3) * rm.volradius
        preds['prim_rot'] = torch.eye(3).to(preds['prim_pos'])[None, None, ...].repeat(bs, self.num_prims, 1, 1)
        preds['prim_scale'] = (1 / scale.reshape(bs, self.num_prims, 1).repeat(1, 1, 3))
        if not self.sample_view:
            preds['Rt'] = torch.Tensor([
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0 * rm.volradius
                ],
                [
                    0.0,
                    -1.0,
                    0.0,
                    0.0 * rm.volradius
                ],
                [
                    0.0,
                    0.0,
                    -1.0,
                    5 * rm.volradius
                ]
                ]).to(prim_volume)[None, ...].repeat(bs, 1, 1)
        else:
            preds['Rt'] = sample_orbit_traj(radius=5*rm.volradius, height=0, start_theta=self.sample_start, end_theta=self.sample_end, num_points=bs).to(prim_volume)
        preds['K'] = torch.Tensor([
            [
                2084.9526697685183,
                0.0,
                512.0
            ],
            [
                0.0,
                2084.9526697685183,
                512.0
            ],
            [
                0.0,
                0.0,
                1.0
            ]]).to(prim_volume)[None, ...].repeat(bs, 1, 1)
        ratio_h = rm.image_height / 1024.
        ratio_w = rm.image_width / 1024.
        preds['K'][:, 0:1, :] *= ratio_h
        preds['K'][:, 1:2, :] *= ratio_w
        rm_preds = rm(
            prim_rgba=preds["prim_rgba"],
            prim_pos=preds["prim_pos"],
            prim_scale=preds["prim_scale"],
            prim_rot=preds["prim_rot"],
            RT=preds["Rt"],
            K=preds["K"],
        )
        rendered_image = rm_preds['rgba_image'].permute(0, 2, 3, 1)[..., :3].contiguous()
        with torch.autocast(device_type='cuda', dtype=precision_dtype, enabled=amp):
            results = self.encoder(rendered_image)
        return results

class ImageMultiViewConditioner(nn.Module):
    def __init__(
        self,
        num_prims,
        dim_feat,
        prim_shape,
        encoder_config,
        sample_view=False,
        view_counts=4,
    ):
        super().__init__()

        self.num_prims = num_prims
        self.dim_feat = dim_feat
        self.prim_shape = prim_shape
        self.view_counts = view_counts
        view_angles = torch.linspace(0.5, 2.5, self.view_counts + 1) * torch.pi
        self.view_angles = view_angles[:-1]
        self.encoder = load_from_config(encoder_config)
    
    def sdf2alpha(self, sdf):
        return torch.exp(-(sdf / 0.005) ** 2)

    @torch.no_grad()
    def forward(self, batch, rm, amp, precision_dtype=torch.float32):
        # TODO: replace with real rendering process in primsdf
        assert 'input_param' in batch, "No parameters in current batch for rendering image conditions"
        prim_volume = batch['input_param']
        bs = prim_volume.shape[0]
        preds = {}
        geo_start_index = 4
        geo_end_index = geo_start_index + self.prim_shape ** 3 # non-inclusive
        tex_start_index = geo_end_index
        tex_end_index = tex_start_index + self.prim_shape ** 3 * 3 # non-inclusive
        feat_geo = prim_volume[:, :, geo_start_index: geo_end_index]
        feat_tex = prim_volume[:, :, tex_start_index: tex_end_index]
        prim_alpha = self.sdf2alpha(feat_geo).reshape(bs, self.num_prims, 1, self.prim_shape, self.prim_shape, self.prim_shape) * 255
        prim_rgb = feat_tex.reshape(bs, self.num_prims, 3, self.prim_shape, self.prim_shape, self.prim_shape) * 255
        preds['prim_rgba'] = torch.concat([prim_rgb, prim_alpha], dim=2)
        pos = prim_volume[:, :, 1:4]
        scale = prim_volume[:, :, 0:1]
        preds['prim_pos'] = pos.reshape(bs, self.num_prims, 3) * rm.volradius
        preds['prim_rot'] = torch.eye(3).to(preds['prim_pos'])[None, None, ...].repeat(bs, self.num_prims, 1, 1)
        preds['prim_scale'] = (1 / scale.reshape(bs, self.num_prims, 1).repeat(1, 1, 3))
        preds['K'] = torch.Tensor([
            [
                2084.9526697685183,
                0.0,
                512.0
            ],
            [
                0.0,
                2084.9526697685183,
                512.0
            ],
            [
                0.0,
                0.0,
                1.0
            ]]).to(prim_volume)[None, ...].repeat(bs, 1, 1)
        ratio_h = rm.image_height / 1024.
        ratio_w = rm.image_width / 1024.
        preds['K'][:, 0:1, :] *= ratio_h
        preds['K'][:, 1:2, :] *= ratio_w
        # we sample view according to view_counts
        cond_list = []
        for view_ang in self.view_angles:
            bs_view_ang = view_ang.repeat(bs,)
            preds['Rt'] = get_pose_on_orbit(radius=5*rm.volradius, height=0, angles=bs_view_ang).to(prim_volume)
            rm_preds = rm(
                prim_rgba=preds["prim_rgba"],
                prim_pos=preds["prim_pos"],
                prim_scale=preds["prim_scale"],
                prim_rot=preds["prim_rot"],
                RT=preds["Rt"],
                K=preds["K"],
            )
            rendered_image = rm_preds['rgba_image'].permute(0, 2, 3, 1)[..., :3].contiguous()
            with torch.autocast(device_type='cuda', dtype=precision_dtype, enabled=amp):
                results = self.encoder(rendered_image)
            cond_list.append(results)
        final_cond = torch.concat(cond_list, dim=1)
        return final_cond

class CLIPImageEncoder(nn.Module):
    def __init__(
        self,
        pretrained_path: str,
        model_spec: str = 'ViT-L-14',
    ):
        super().__init__()

        self.model, _, _ = open_clip.create_model_and_transforms(model_spec, pretrained=pretrained_path)
        self.model_resolution = self.model.visual.image_size
        self.preprocess = Compose([
            Resize(self.model_resolution, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(self.model_resolution),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.model.eval()
        # self.tokenizer = open_clip.get_tokenizer(model_spec)

    @torch.no_grad()
    def forward(self, img):
        assert img.shape[-1] == 3
        img = img.permute(0, 3, 1, 2) / 255.
        image = self.preprocess(img)
        image_features = self.model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

class CLIPImageTokenEncoder(nn.Module):
    def __init__(
        self,
        pretrained_path: str,
        model_spec: str = 'ViT-L-14',
    ):
        super().__init__()

        self.model, _, _ = open_clip.create_model_and_transforms(model_spec, pretrained=pretrained_path)
        self.model.visual.output_tokens = True
        self.model_resolution = self.model.visual.image_size
        self.preprocess = Compose([
            Resize(self.model_resolution, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(self.model_resolution),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.model.eval()

    @torch.no_grad()
    def forward(self, img):
        assert img.shape[-1] == 3
        img = img.permute(0, 3, 1, 2) / 255.
        image = self.preprocess(img)
        _, image_tokens = self.model.encode_image(image)
        # [B, T, D] - [B, 256, 1024]
        image_tokens /= image_tokens.norm(dim=-1, keepdim=True)
        return image_tokens