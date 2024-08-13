import cv2
import os
import numpy as np
import torch
import imageio
from torchvision.utils import make_grid, save_image
from .ray_marcher import RayMarcher, generate_colored_boxes

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

def render_mvp_boxes(rm, batch, preds):
    with torch.no_grad():
        boxes_rgba = generate_colored_boxes(
            preds["prim_rgba"],
            preds["prim_rot"],
        )
        preds_boxes = rm(
            prim_rgba=boxes_rgba,
            prim_pos=preds["prim_pos"],
            prim_scale=preds["prim_scale"],
            prim_rot=preds["prim_rot"],
            RT=batch["Rt"],
            K=batch["K"],
        )

    return preds_boxes["rgba_image"][:, :3].permute(0, 2, 3, 1)


def save_image_summary(path, batch, preds):
    rgb = preds["rgb"].detach().permute(0, 3, 1, 2)
    # rgb_gt = batch["image"]
    rgb_boxes = preds["rgb_boxes"].detach().permute(0, 3, 1, 2)
    bs = rgb_boxes.shape[0]
    if "folder" in batch and "key" in batch:
        obj_list = []
        for bs_idx in range(bs):
            tmp_img = rgb_boxes[bs_idx].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
            tmp_img = np.ascontiguousarray(tmp_img)
            folder = batch['folder'][bs_idx]
            key = batch['key'][bs_idx]
            obj_list.append("{}/{}\n".format(folder, key))
            cv2.putText(tmp_img, "{}".format(folder), (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
            cv2.putText(tmp_img, "{}".format(key), (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
            tmp_img_torch = torch.as_tensor(tmp_img).permute(2, 0, 1).float()
            rgb_boxes[bs_idx] = tmp_img_torch
        with open(os.path.splitext(path)[0]+".txt", "w") as f:
            f.writelines(obj_list)
    img = make_grid(torch.cat([rgb, rgb_boxes], dim=2) / 255.0).clip(0.0, 1.0)
    save_image(img, path)


@torch.no_grad()
def visualize_primsdf_box(image_save_path, model, rm: RayMarcher, device):
    # prim_rgba: primitive payload [B, K, 4, S, S, S],
    # K - # of primitives, S - primitive size
    # prim_pos: locations [B, K, 3]
    # prim_rot: rotations [B, K, 3, 3]
    # prim_scale: scales [B, K, 3]
    # K: intrinsics [B, 3, 3]
    # RT: extrinsics [B, 3, 4]
    preds = {}
    batch = {}
    prim_alpha = model.sdf2alpha(model.feat_geo).reshape(1, model.num_prims, 1, model.prim_shape, model.prim_shape, model.prim_shape) * 255
    prim_rgb = model.feat_tex.reshape(1, model.num_prims, 3, model.prim_shape, model.prim_shape, model.prim_shape) * 255
    preds['prim_rgba'] = torch.concat([prim_rgb, prim_alpha], dim=2)
    preds['prim_pos'] = model.pos.reshape(1, model.num_prims, 3) * rm.volradius
    preds['prim_rot'] = torch.eye(3).to(preds['prim_pos'])[None, None, ...].repeat(1, model.num_prims, 1, 1)
    preds['prim_scale'] = (1 / model.scale.reshape(1, model.num_prims, 1).repeat(1, 1, 3))
    batch['Rt'] = torch.Tensor([
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
        ]).to(device)[None, ...]
    batch['K'] = torch.Tensor([
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
            ]]).to(device)[None, ...]
    ratio_h = rm.image_height / 1024.
    ratio_w = rm.image_width / 1024.
    batch['K'][:, 0:1, :] *= ratio_h
    batch['K'][:, 1:2, :] *= ratio_w
    # raymarcher is in mm
    rm_preds = rm(
        prim_rgba=preds["prim_rgba"],
        prim_pos=preds["prim_pos"],
        prim_scale=preds["prim_scale"],
        prim_rot=preds["prim_rot"],
        RT=batch["Rt"],
        K=batch["K"],
    )
    rgba = rm_preds["rgba_image"].permute(0, 2, 3, 1)
    preds.update(alpha=rgba[..., -1].contiguous(), rgb=rgba[..., :3].contiguous())
    with torch.no_grad():
        preds["rgb_boxes"] = render_mvp_boxes(rm, batch, preds)
    save_image_summary(image_save_path, batch, preds)

@torch.no_grad()
def render_primsdf(image_save_path, model, rm, device):
    preds = {}
    batch = {}
    preds['prim_pos'] = model.pos.reshape(1, model.num_prims, 3) * rm.volradius
    preds['prim_rot'] = torch.eye(3).to(preds['prim_pos'])[None, None, ...].repeat(1, model.num_prims, 1, 1)
    preds['prim_scale'] = (1 / model.scale.reshape(1, model.num_prims, 1).repeat(1, 1, 3))
    batch['Rt'] = torch.Tensor([
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
        ]).to(device)[None, ...]
    batch['K'] = torch.Tensor([
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
            ]]).to(device)[None, ...]
    ratio_h = rm.image_height / 1024.
    ratio_w = rm.image_width / 1024.
    batch['K'][:, 0:1, :] *= ratio_h
    batch['K'][:, 1:2, :] *= ratio_w
    # test rendering
    all_sampled_sdf = []
    all_sampled_tex = []
    for i in range(model.prim_shape ** 3):
        with torch.no_grad():
            model_prediction = model(model.sdf_sampled_point[:, i, :].to(device))
            sampled_sdf = model_prediction['sdf']
            sampled_rgb = model_prediction['tex']
            all_sampled_sdf.append(sampled_sdf)
            all_sampled_tex.append(sampled_rgb)
    sampled_sdf = torch.stack(all_sampled_sdf, dim=1)
    sampled_tex = torch.stack(all_sampled_tex, dim=1).permute(0, 2, 1).reshape(1, model.num_prims, 3, model.prim_shape, model.prim_shape, model.prim_shape) * 255
    prim_rgb = sampled_tex
    prim_alpha = model.sdf2alpha(sampled_sdf).reshape(1, model.num_prims, 1, model.prim_shape, model.prim_shape, model.prim_shape) * 255
    preds['prim_rgba'] = torch.concat([prim_rgb, prim_alpha], dim=2)
    rm_preds = rm(
        prim_rgba=preds["prim_rgba"],
        prim_pos=preds["prim_pos"],
        prim_scale=preds["prim_scale"],
        prim_rot=preds["prim_rot"],
        RT=batch["Rt"],
        K=batch["K"],
    )

    rgba = rm_preds["rgba_image"].permute(0, 2, 3, 1)
    preds.update(alpha=rgba[..., -1].contiguous(), rgb=rgba[..., :3].contiguous())
    with torch.no_grad():
        preds["rgb_boxes"] = render_mvp_boxes(rm, batch, preds)
    save_image_summary(image_save_path, batch, preds)

@torch.no_grad()
def visualize_primvolume(image_save_path, batch, prim_volume, rm: RayMarcher, device):
    # prim_volume - [B, nprims, 4+6*8^3]
    def sdf2alpha(sdf):
        return torch.exp(-(sdf / 0.005) ** 2)
    preds = {}
    prim_shape = int(np.round(((prim_volume.shape[2] - 4) / 6) ** (1/3)))
    num_prims = prim_volume.shape[1]
    bs = prim_volume.shape[0]
    geo_start_index = 4
    geo_end_index = geo_start_index + prim_shape ** 3 # non-inclusive
    tex_start_index = geo_end_index
    tex_end_index = tex_start_index + prim_shape ** 3 * 3 # non-inclusive
    mat_start_index = tex_end_index
    mat_end_index = mat_start_index + prim_shape ** 3 * 2

    feat_geo = prim_volume[:, :, geo_start_index: geo_end_index]
    feat_tex = prim_volume[:, :, tex_start_index: tex_end_index]
    prim_alpha = sdf2alpha(feat_geo).reshape(bs, num_prims, 1, prim_shape, prim_shape, prim_shape) * 255
    prim_rgb = feat_tex.reshape(bs, num_prims, 3, prim_shape, prim_shape, prim_shape) * 255
    preds['prim_rgba'] = torch.concat([prim_rgb, prim_alpha], dim=2)
    pos = prim_volume[:, :, 1:4]
    scale = prim_volume[:, :, 0:1]
    preds['prim_pos'] = pos.reshape(bs, num_prims, 3) * rm.volradius
    preds['prim_rot'] = torch.eye(3).to(preds['prim_pos'])[None, None, ...].repeat(bs, num_prims, 1, 1)
    preds['prim_scale'] = (1 / scale.reshape(bs, num_prims, 1).repeat(1, 1, 3))
    batch['Rt'] = torch.Tensor([
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
        ]).to(device)[None, ...].repeat(bs, 1, 1)
    batch['K'] = torch.Tensor([
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
            ]]).to(device)[None, ...].repeat(bs, 1, 1)
    ratio_h = rm.image_height / 1024.
    ratio_w = rm.image_width / 1024.
    batch['K'][:, 0:1, :] *= ratio_h
    batch['K'][:, 1:2, :] *= ratio_w
    # raymarcher is in mm
    rm_preds = rm(
        prim_rgba=preds["prim_rgba"],
        prim_pos=preds["prim_pos"],
        prim_scale=preds["prim_scale"],
        prim_rot=preds["prim_rot"],
        RT=batch["Rt"],
        K=batch["K"],
    )
    rgba = rm_preds["rgba_image"].permute(0, 2, 3, 1)
    preds.update(alpha=rgba[..., -1].contiguous(), rgb=rgba[..., :3].contiguous())
    with torch.no_grad():
        preds["rgb_boxes"] = render_mvp_boxes(rm, batch, preds)
    save_image_summary(image_save_path, batch, preds)

@torch.no_grad()
def visualize_multiview_primvolume(image_save_path, batch, prim_volume, view_counts, rm: RayMarcher, device):
    # prim_volume - [B, nprims, 4+6*8^3]
    view_angles = torch.linspace(0.5, 2.5, view_counts + 1) * torch.pi
    view_angles = view_angles[:-1]
    def sdf2alpha(sdf):
        return torch.exp(-(sdf / 0.005) ** 2)
    preds = {}
    prim_shape = int(np.round(((prim_volume.shape[2] - 4) / 6) ** (1/3)))
    num_prims = prim_volume.shape[1]
    bs = prim_volume.shape[0]
    geo_start_index = 4
    geo_end_index = geo_start_index + prim_shape ** 3 # non-inclusive
    tex_start_index = geo_end_index
    tex_end_index = tex_start_index + prim_shape ** 3 * 3 # non-inclusive
    mat_start_index = tex_end_index
    mat_end_index = mat_start_index + prim_shape ** 3 * 2

    feat_geo = prim_volume[:, :, geo_start_index: geo_end_index]
    feat_tex = prim_volume[:, :, tex_start_index: tex_end_index]
    prim_alpha = sdf2alpha(feat_geo).reshape(bs, num_prims, 1, prim_shape, prim_shape, prim_shape) * 255
    prim_rgb = feat_tex.reshape(bs, num_prims, 3, prim_shape, prim_shape, prim_shape) * 255
    preds['prim_rgba'] = torch.concat([prim_rgb, prim_alpha], dim=2)
    pos = prim_volume[:, :, 1:4]
    scale = prim_volume[:, :, 0:1]
    preds['prim_pos'] = pos.reshape(bs, num_prims, 3) * rm.volradius
    preds['prim_rot'] = torch.eye(3).to(preds['prim_pos'])[None, None, ...].repeat(bs, num_prims, 1, 1)
    preds['prim_scale'] = (1 / scale.reshape(bs, num_prims, 1).repeat(1, 1, 3))
    batch['K'] = torch.Tensor([
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
            ]]).to(device)[None, ...].repeat(bs, 1, 1)
    ratio_h = rm.image_height / 1024.
    ratio_w = rm.image_width / 1024.
    batch['K'][:, 0:1, :] *= ratio_h
    batch['K'][:, 1:2, :] *= ratio_w

    final_preds = {}
    final_preds['rgb'] = []
    final_preds['rgb_boxes'] = []
    for view_ang in view_angles:
        bs_view_ang = view_ang.repeat(bs,)
        batch['Rt'] = get_pose_on_orbit(radius=5*rm.volradius, height=0, angles=bs_view_ang).to(prim_volume)
        # raymarcher is in mm
        rm_preds = rm(
            prim_rgba=preds["prim_rgba"],
            prim_pos=preds["prim_pos"],
            prim_scale=preds["prim_scale"],
            prim_rot=preds["prim_rot"],
            RT=batch["Rt"],
            K=batch["K"],
        )
        rgba = rm_preds["rgba_image"].permute(0, 2, 3, 1)
        preds.update(alpha=rgba[..., -1].contiguous(), rgb=rgba[..., :3].contiguous())
        with torch.no_grad():
            preds["rgb_boxes"] = render_mvp_boxes(rm, batch, preds)
        final_preds['rgb'].append(preds['rgb'])
        final_preds['rgb_boxes'].append(preds['rgb_boxes'])
    final_preds['rgb'] = torch.concat(final_preds['rgb'], dim=0)
    final_preds['rgb_boxes'] = torch.concat(final_preds['rgb_boxes'], dim=0)
    save_image_summary(image_save_path, batch, final_preds)


@torch.no_grad()
def visualize_video_primvolume(video_save_folder, batch, prim_volume, view_counts, rm: RayMarcher, device):
    # prim_volume - [B, nprims, 4+6*8^3]
    view_angles = torch.linspace(1.5, 3.5, view_counts + 1) * torch.pi
    def sdf2alpha(sdf):
        return torch.exp(-(sdf / 0.005) ** 2)
    preds = {}
    prim_shape = int(np.round(((prim_volume.shape[2] - 4) / 6) ** (1/3)))
    num_prims = prim_volume.shape[1]
    bs = prim_volume.shape[0]
    geo_start_index = 4
    geo_end_index = geo_start_index + prim_shape ** 3 # non-inclusive
    tex_start_index = geo_end_index
    tex_end_index = tex_start_index + prim_shape ** 3 * 3 # non-inclusive
    mat_start_index = tex_end_index
    mat_end_index = mat_start_index + prim_shape ** 3 * 2

    feat_geo = prim_volume[:, :, geo_start_index: geo_end_index]
    feat_tex = prim_volume[:, :, tex_start_index: tex_end_index]
    prim_alpha = sdf2alpha(feat_geo).reshape(bs, num_prims, 1, prim_shape, prim_shape, prim_shape) * 255
    prim_rgb = feat_tex.reshape(bs, num_prims, 3, prim_shape, prim_shape, prim_shape) * 255
    preds['prim_rgba'] = torch.concat([prim_rgb, prim_alpha], dim=2)
    pos = prim_volume[:, :, 1:4]
    scale = prim_volume[:, :, 0:1]
    preds['prim_pos'] = pos.reshape(bs, num_prims, 3) * rm.volradius
    preds['prim_rot'] = torch.eye(3).to(preds['prim_pos'])[None, None, ...].repeat(bs, num_prims, 1, 1)
    preds['prim_scale'] = (1 / scale.reshape(bs, num_prims, 1).repeat(1, 1, 3))
    batch['K'] = torch.Tensor([
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
            ]]).to(device)[None, ...].repeat(bs, 1, 1)
    ratio_h = rm.image_height / 1024.
    ratio_w = rm.image_width / 1024.
    batch['K'][:, 0:1, :] *= ratio_h
    batch['K'][:, 1:2, :] *= ratio_w

    final_preds = {}
    final_preds['rgb'] = []
    final_preds['rgb_boxes'] = []
    for view_ang in view_angles:
        bs_view_ang = view_ang.repeat(bs,)
        batch['Rt'] = get_pose_on_orbit(radius=5*rm.volradius, height=0, angles=bs_view_ang).to(prim_volume)
        # raymarcher is in mm
        rm_preds = rm(
            prim_rgba=preds["prim_rgba"],
            prim_pos=preds["prim_pos"],
            prim_scale=preds["prim_scale"],
            prim_rot=preds["prim_rot"],
            RT=batch["Rt"],
            K=batch["K"],
        )
        rgba = rm_preds["rgba_image"].permute(0, 2, 3, 1)
        preds.update(alpha=rgba[..., -1].contiguous(), rgb=rgba[..., :3].contiguous())
        with torch.no_grad():
            preds["rgb_boxes"] = render_mvp_boxes(rm, batch, preds)
        final_preds['rgb'].append(preds['rgb'])
        final_preds['rgb_boxes'].append(preds['rgb_boxes'])

    assert len(final_preds['rgb']) == len(final_preds['rgb_boxes'])
    final_preds['rgb'] = torch.concat(final_preds['rgb'], dim=0)
    final_preds['rgb_boxes'] = torch.concat(final_preds['rgb_boxes'], dim=0)
    total_num_frames = final_preds['rgb'].shape[0]
    rgb_video = os.path.join(video_save_folder, 'rgb.mp4')
    rgb_video_out = imageio.get_writer(rgb_video, fps=20)
    prim_video = os.path.join(video_save_folder, 'prim.mp4')
    prim_video_out = imageio.get_writer(prim_video, fps=20)

    rgb_np = np.clip(final_preds['rgb'].detach().cpu().numpy(), 0, 255).astype(np.uint8)
    prim_np = np.clip(final_preds['rgb_boxes'].detach().cpu().numpy(), 0, 255).astype(np.uint8)
    for fidx in range(total_num_frames):
        rgb_video_out.append_data(rgb_np[fidx])
        prim_video_out.append_data(prim_np[fidx])
    rgb_video_out.close()
    prim_video_out.close()