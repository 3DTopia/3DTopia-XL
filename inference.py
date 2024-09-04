import os
import sys
import io

import torch
import numpy as np
from omegaconf import OmegaConf
import PIL.Image
from PIL import Image
import rembg

from dva.ray_marcher import RayMarcher
from dva.io import load_from_config
from dva.utils import to_device
from dva.visualize import visualize_primvolume, visualize_video_primvolume
from models.diffusion import create_diffusion
import logging
from tqdm import tqdm

import mcubes
import xatlas
import nvdiffrast.torch as dr
import cv2
from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.neighbors import NearestNeighbors
from utils.meshutils import clean_mesh, decimate_mesh
from utils.mesh import Mesh
from utils.uv_unwrap import box_projection_uv_unwrap, compute_vertex_normal
logger = logging.getLogger("inference.py")

glctx = dr.RasterizeCudaContext()

def remove_background(image: PIL.Image.Image,
    rembg_session = None,
    force: bool = False,
    **rembg_kwargs,
) -> PIL.Image.Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image

def resize_foreground(
    image: PIL.Image.Image,
    ratio: float,
) -> PIL.Image.Image:
    image = np.array(image)
    assert image.shape[-1] == 4
    alpha = np.where(image[..., 3] > 0)
    y1, y2, x1, x2 = (
        alpha[0].min(),
        alpha[0].max(),
        alpha[1].min(),
        alpha[1].max(),
    )
    # crop the foreground
    fg = image[y1:y2, x1:x2]
    # pad to square
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(
        fg,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )

    # compute padding according to the ratio
    new_size = int(new_image.shape[0] / ratio)
    # pad to size, double side
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(
        new_image,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )
    new_image = PIL.Image.fromarray(new_image)
    return new_image

def extract_texmesh(args, model, output_path, device):
    # Prepare directory
    ins_dir = output_path
    # Noise Filter
    raw_srt_param = model.srt_param.clone()
    raw_feat_param = model.feat_param.clone()
    prim_position = raw_srt_param[:, 1:4]
    prim_scale = raw_srt_param[:, 0:1]
    dist = torch.sqrt(torch.sum((prim_position[:, None, :] - prim_position[None, :, :]) ** 2, dim=-1))
    dist += torch.eye(prim_position.shape[0]).to(raw_srt_param)
    min_dist, min_indices = dist.min(1)
    dst_prim_scale = prim_scale[min_indices, :]
    min_scale_converage = prim_scale * 1. + dst_prim_scale * 1.
    prim_mask = min_dist < min_scale_converage[:, 0]
    filtered_srt_param = raw_srt_param[prim_mask, :]
    filtered_feat_param = raw_feat_param[prim_mask, ...]
    model.srt_param.data = filtered_srt_param
    model.feat_param.data = filtered_feat_param
    print(f'[INFO] Mesh Extraction on PrimX: srt={model.srt_param.shape} feat={model.feat_param.shape}')

    # Get SDFs
    with torch.no_grad():
        xx = torch.linspace(-1, 1, args.mc_resolution, device=device)
        pts = torch.stack(torch.meshgrid(xx, xx, xx, indexing='ij'), dim=-1).reshape(-1,3)
        chunks = torch.split(pts, args.batch_size)
        dists = []
        for chunk_pts in tqdm(chunks):
            preds = model(chunk_pts)
            dists.append(preds['sdf'].detach())
    dists = torch.cat(dists, dim=0)
    grid = dists.reshape(args.mc_resolution, args.mc_resolution, args.mc_resolution)

    # Meshify
    vertices, triangles = mcubes.marching_cubes(grid.cpu().numpy(), 0.0)

    # Resize + recenter
    b_min_np = np.array([-1., -1., -1.])
    b_max_np = np.array([ 1.,  1.,  1.])
    vertices = vertices / (args.mc_resolution - 1.0) * (b_max_np - b_min_np) + b_min_np

    vertices, triangles = clean_mesh(vertices, triangles, min_f=8, min_d=5, repair=True, remesh=False)

    if args.decimate > 0 and triangles.shape[0] > args.decimate:
        vertices, triangles = decimate_mesh(vertices, triangles, args.decimate, remesh=args.remesh)

    h0 = 1024
    w0 = 1024
    ssaa = 1
    fp16 = True
    v_np = vertices.astype(np.float32)
    f_np = triangles.astype(np.int64)
    v = torch.from_numpy(vertices).float().contiguous().to(device)
    f = torch.from_numpy(triangles.astype(np.int64)).to(torch.int64).contiguous().to(device)

    if args.fast_unwrap:
        print(f'[INFO] running box-based fast unwrapping to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')
        v_normal = compute_vertex_normal(v, f)
        uv, indices = box_projection_uv_unwrap(v, v_normal, f, 0.02)
        indv_v = v[f].reshape(-1, 3)
        indv_faces = torch.arange(indv_v.shape[0], device=device, dtype=f.dtype).reshape(-1, 3)
        uv_flat = uv[indices].reshape((-1, 2))
        v = indv_v.contiguous()
        f = indv_faces.contiguous()
        ft_np = f.cpu().numpy()
        vt_np = uv_flat.cpu().numpy()
    else:
        print(f'[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')
        # unwrap uv in contracted space
        atlas = xatlas.Atlas()
        atlas.add_mesh(v_np, f_np)
        chart_options = xatlas.ChartOptions()
        chart_options.max_iterations = 0 # disable merge_chart for faster unwrap...
        pack_options = xatlas.PackOptions()
        atlas.generate(chart_options=chart_options, pack_options=pack_options)
        _, ft_np, vt_np = atlas[0] # [N], [M, 3], [N, 2]
    vt = torch.from_numpy(vt_np.astype(np.float32)).float().contiguous().to(device)
    ft = torch.from_numpy(ft_np.astype(np.int64)).int().contiguous().to(device)
    uv = vt * 2.0 - 1.0 # uvs to range [-1, 1]
    uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1) # [N, 4]

    if ssaa > 1:
        h = int(h0 * ssaa)
        w = int(w0 * ssaa)
    else:
        h, w = h0, w0

    rast, _ = dr.rasterize(glctx, uv.unsqueeze(0), ft, (h, w)) # [1, h, w, 4]
    xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f.int()) # [1, h, w, 3]
    mask, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f.int()) # [1, h, w, 1]
    # masked query 
    xyzs = xyzs.view(-1, 3)
    mask = (mask > 0).view(-1)
    feats = torch.zeros(h * w, 6, device=device, dtype=torch.float32)

    if mask.any():
        xyzs = xyzs[mask] # [M, 3]
        # batched inference to avoid OOM
        all_feats = []
        head = 0
        chunk_size = args.batch_size
        while head < xyzs.shape[0]:
            tail = min(head + chunk_size, xyzs.shape[0])
            with torch.cuda.amp.autocast(enabled=fp16):
                preds = model(xyzs[head:tail])
                # [R, G, B, NA, roughness, metallic]
                all_feats.append(torch.concat([preds['tex'].float(), torch.zeros_like(preds['tex'])[..., 0:1].float(), preds['mat'].float()], dim=-1))
            head += chunk_size
        feats[mask] = torch.cat(all_feats, dim=0)
    feats = feats.view(h, w, -1) # 6 channels
    mask = mask.view(h, w)
    # quantize [0.0, 1.0] to [0, 255]
    feats = feats.cpu().numpy()
    feats = (feats * 255)

    ### NN search as a queer antialiasing ...
    mask = mask.cpu().numpy()
    inpaint_region = binary_dilation(mask, iterations=32) # pad width
    inpaint_region[mask] = 0
    search_region = mask.copy()
    not_search_region = binary_erosion(search_region, iterations=3)
    search_region[not_search_region] = 0
    search_coords = np.stack(np.nonzero(search_region), axis=-1)
    inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)
    knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
    _, indices = knn.kneighbors(inpaint_coords)
    feats[tuple(inpaint_coords.T)] = feats[tuple(search_coords[indices[:, 0]].T)]
    # do ssaa after the NN search, in numpy
    feats0 = cv2.cvtColor(feats[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR) # albedo
    feats1 = cv2.cvtColor(feats[..., 3:].astype(np.uint8), cv2.COLOR_RGB2BGR) # visibility features
    if ssaa > 1:
        feats0 = cv2.resize(feats0, (w0, h0), interpolation=cv2.INTER_LINEAR)
        feats1 = cv2.resize(feats1, (w0, h0), interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(os.path.join(ins_dir, f'texture.jpg'), feats0)
    cv2.imwrite(os.path.join(ins_dir, f'roughness_metallic.jpg'), feats1)

    target_mesh = Mesh(v=torch.from_numpy(v_np).contiguous(), f=torch.from_numpy(f_np).contiguous(), ft=ft.contiguous(), vt=torch.from_numpy(vt_np).contiguous(), albedo=torch.from_numpy(feats[..., :3]) / 255, metallicRoughness=torch.from_numpy(feats[..., 3:]) / 255)
    target_mesh.write(os.path.join(ins_dir, f'pbr_mesh.glb'))
    model.srt_param.data = raw_srt_param
    model.feat_param.data = raw_feat_param

def main(config):
    logging.basicConfig(level=logging.INFO)
    ddim_steps = config.inference.ddim
    if ddim_steps > 0:
        use_ddim = True
    else:
        use_ddim = False
    cfg_scale = config.inference.get("cfg", 0.0)

    inference_dir = f"{config.output_dir}/inference_folder"
    os.makedirs(inference_dir, exist_ok=True)

    amp = False
    precision = config.inference.get("precision", 'fp16')
    if precision == 'tf32':
        precision_dtype = torch.float32
    elif precision == 'fp16':
        amp = True
        precision_dtype = torch.float16
    else:
       raise NotImplementedError("{} precision is not supported".format(precision))

    device = torch.device(f"cuda:{0}")
    seed = config.inference.seed
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    
    model = load_from_config(config.model.generator)
    vae = load_from_config(config.model.vae)
    conditioner = load_from_config(config.model.conditioner)
    vae_state_dict = torch.load(config.model.vae_checkpoint_path, map_location='cpu')
    vae.load_state_dict(vae_state_dict['model_state_dict'])
    
    if config.checkpoint_path:
        state_dict = torch.load(config.checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict['ema'])
    vae = vae.to(device)
    conditioner = conditioner.to(device)
    model = model.to(device)
    config.diffusion.pop("timestep_respacing")
    if use_ddim:
        respacing = "ddim{}".format(ddim_steps)
    else:
        respacing = ""
    diffusion = create_diffusion(timestep_respacing=respacing, **config.diffusion)  # default: 1000 steps, linear noise schedule
    if use_ddim:
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.p_sample_loop_progressive
    
    if cfg_scale > 0:
        fwd_fn = model.forward_with_cfg
    else:
        fwd_fn = model.forward

    rm = RayMarcher(
        config.image_height,
        config.image_width,
        **config.rm,
    ).to(device)

    perchannel_norm = False
    if "latent_mean" in config.model:
        latent_mean = torch.Tensor(config.model.latent_mean)[None, None, :].to(device)
        latent_std = torch.Tensor(config.model.latent_std)[None, None, :].to(device)
        assert latent_mean.shape[-1] == config.model.generator.in_channels
        perchannel_norm = True

    model.eval()
    examples_dir = config.inference.input_dir
    img_list = os.listdir(examples_dir)
    rembg_session = rembg.new_session()
    logger.info(f"Starting Inference...")
    for img_path in img_list:
        full_img_path = os.path.join(examples_dir, img_path)
        img_name = img_path[:-4]
        current_output_dir = os.path.join(inference_dir, img_name)
        os.makedirs(current_output_dir, exist_ok=True)
        input_image = Image.open(full_img_path)
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)
        raw_image = np.array(input_image)
        mask = (raw_image[..., -1][..., None] > 0) * 1
        raw_image = raw_image[..., :3] * mask
        input_cond = torch.from_numpy(np.array(raw_image)[None, ...]).to(device)
        with torch.no_grad():
            latent = torch.randn(1, config.model.num_prims, 1, 4, 4, 4)
            batch = {}
            inf_bs = 1
            inf_x = torch.randn(inf_bs, config.model.num_prims, 68).to(device)
            y = conditioner.encoder(input_cond)
            model_kwargs = dict(y=y[:inf_bs, ...], precision_dtype=precision_dtype, enable_amp=amp)
            if cfg_scale > 0:
                model_kwargs['cfg_scale'] = cfg_scale
            sampled_count = -1
            for samples in sample_fn(fwd_fn, inf_x.shape, inf_x, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            ):
                sampled_count += 1
                if not (sampled_count % 10 == 0 or sampled_count == diffusion.num_timesteps - 1):
                    continue
                else:
                    recon_param = samples["sample"].reshape(inf_bs, config.model.num_prims, -1)
                    if perchannel_norm:
                        recon_param = recon_param / config.model.latent_nf * latent_std + latent_mean
                    recon_srt_param = recon_param[:, :, 0:4]
                    recon_feat_param = recon_param[:, :, 4:] # [8, 2048, 64]
                    recon_feat_param_list = []
                    # one-by-one to avoid oom
                    for inf_bidx in range(inf_bs):
                        if not perchannel_norm:
                            decoded = vae.decode(recon_feat_param[inf_bidx, ...].reshape(1*config.model.num_prims, *latent.shape[-4:]) / config.model.latent_nf)
                        else:
                            decoded = vae.decode(recon_feat_param[inf_bidx, ...].reshape(1*config.model.num_prims, *latent.shape[-4:]))
                        recon_feat_param_list.append(decoded.detach())
                    recon_feat_param = torch.concat(recon_feat_param_list, dim=0)
                    # invert normalization
                    if not perchannel_norm:
                        recon_srt_param[:, :, 0:1] = (recon_srt_param[:, :, 0:1] / 10) + 0.05
                    recon_feat_param[:, 0:1, ...] /= 5.
                    recon_feat_param[:, 1:, ...] = (recon_feat_param[:, 1:, ...] + 1) / 2.
                    recon_feat_param = recon_feat_param.reshape(inf_bs, config.model.num_prims, -1)
                    recon_param = torch.concat([recon_srt_param, recon_feat_param], dim=-1)
                    visualize_primvolume("{}/dstep{:04d}_recon.jpg".format(current_output_dir, sampled_count), batch, recon_param, rm, device)
            visualize_video_primvolume(current_output_dir, batch, recon_param, 60, rm, device)
            prim_params = {'srt_param': recon_srt_param[0].detach().cpu(), 'feat_param': recon_feat_param[0].detach().cpu()}
            torch.save({'model_state_dict': prim_params}, "{}/denoised.pt".format(current_output_dir))

    if config.inference.export_glb:
        logger.info(f"Starting GLB Mesh Extraction...")
        config.model.pop("vae")
        config.model.pop("vae_checkpoint_path")
        config.model.pop("conditioner")
        config.model.pop("generator")
        config.model.pop("latent_nf")
        config.model.pop("latent_mean")
        config.model.pop("latent_std")
        model_primx = load_from_config(config.model)
        for img_path in img_list:
            img_name = img_path[:-4]
            output_path = os.path.join(inference_dir, img_name)
            denoise_param_path = os.path.join(inference_dir, img_name, 'denoised.pt')
            ckpt_weight = torch.load(denoise_param_path, map_location='cpu')['model_state_dict']
            model_primx.load_state_dict(ckpt_weight)
            model_primx.to(device)
            model_primx.eval()
            with torch.no_grad():
                model_primx.srt_param[:, 1:4] *= 0.85
                extract_texmesh(config.inference, model_primx, output_path, device)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    # manually enable tf32 to get speedup on A100 GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # set config
    config = OmegaConf.load(str(sys.argv[1]))
    config_cli = OmegaConf.from_cli(args_list=sys.argv[2:])
    if config_cli:
        logger.info("overriding with following values from args:")
        logger.info(OmegaConf.to_yaml(config_cli))
        config = OmegaConf.merge(config, config_cli)

    main(config)
