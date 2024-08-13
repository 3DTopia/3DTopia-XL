import os
import imageio
import numpy as np

os.system("bash install.sh")

from omegaconf import OmegaConf
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import rembg
import gradio as gr
from gradio_litmodel3d import LitModel3D
from dva.io import load_from_config
from dva.ray_marcher import RayMarcher
from dva.visualize import visualize_primvolume, visualize_video_primvolume
from inference import remove_background, resize_foreground, extract_texmesh
from models.diffusion import create_diffusion
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download(repo_id="frozenburning/3DTopia-XL", filename="model_sview_dit_fp16.pt")
vae_ckpt_path = hf_hub_download(repo_id="frozenburning/3DTopia-XL", filename="model_vae_fp16.pt")

GRADIO_PRIM_VIDEO_PATH = 'prim.mp4'
GRADIO_RGB_VIDEO_PATH = 'rgb.mp4'
GRADIO_MAT_VIDEO_PATH = 'mat.mp4'
GRADIO_GLB_PATH = 'pbr_mesh.glb'
CONFIG_PATH = "./configs/inference_dit.yml"

config = OmegaConf.load(CONFIG_PATH)
config.checkpoint_path = ckpt_path
config.model.vae_checkpoint_path = vae_ckpt_path
# model
model = load_from_config(config.model.generator)
state_dict = torch.load(config.checkpoint_path, map_location='cpu')
model.load_state_dict(state_dict['ema'])
vae = load_from_config(config.model.vae)
vae_state_dict = torch.load(config.model.vae_checkpoint_path, map_location='cpu')
vae.load_state_dict(vae_state_dict['model_state_dict'])
conditioner = load_from_config(config.model.conditioner)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = vae.to(device)
conditioner = conditioner.to(device)
model = model.to(device)
model.eval()

amp = True
precision_dtype = torch.float16

rm = RayMarcher(
    256,
    256,
    **config.rm,
).to(device)

perchannel_norm = False
if "latent_mean" in config.model:
    latent_mean = torch.Tensor(config.model.latent_mean)[None, None, :].to(device)
    latent_std = torch.Tensor(config.model.latent_std)[None, None, :].to(device)
    assert latent_mean.shape[-1] == config.model.generator.in_channels
    perchannel_norm = True
latent_nf = config.model.latent_nf

config.diffusion.pop("timestep_respacing")
config.model.pop("vae")
config.model.pop("vae_checkpoint_path")
config.model.pop("conditioner")
config.model.pop("generator")
config.model.pop("latent_nf")
config.model.pop("latent_mean")
config.model.pop("latent_std")
model_primx = load_from_config(config.model)
# load rembg
rembg_session = rembg.new_session()

# background removal function
def background_remove_process(input_image):
    input_image = remove_background(input_image, rembg_session)
    input_image = resize_foreground(input_image, 0.85)
    input_cond_preview_pil = input_image
    raw_image = np.array(input_image)
    mask = (raw_image[..., -1][..., None] > 0) * 1
    raw_image = raw_image[..., :3] * mask
    input_cond = torch.from_numpy(np.array(raw_image)[None, ...]).to(device)
    return gr.update(interactive=True), input_cond, input_cond_preview_pil

# process function
def process(input_cond, input_num_steps, input_seed=42, input_cfg=6.0):
    # seed
    torch.manual_seed(input_seed)

    os.makedirs(config.output_dir, exist_ok=True)
    output_rgb_video_path = os.path.join(config.output_dir, GRADIO_RGB_VIDEO_PATH)
    output_prim_video_path = os.path.join(config.output_dir, GRADIO_PRIM_VIDEO_PATH)
    output_mat_video_path = os.path.join(config.output_dir, GRADIO_MAT_VIDEO_PATH)

    respacing = "ddim{}".format(input_num_steps)
    diffusion = create_diffusion(timestep_respacing=respacing, **config.diffusion)
    sample_fn = diffusion.ddim_sample_loop_progressive
    fwd_fn = model.forward_with_cfg

    # text-conditioned
    if input_cond is None:
        raise NotImplementedError
    
    with torch.no_grad():
        latent = torch.randn(1, config.model.num_prims, 1, 4, 4, 4)
        batch = {}
        inf_bs = 1
        inf_x = torch.randn(inf_bs, config.model.num_prims, 68).to(device)
        y = conditioner.encoder(input_cond)
        model_kwargs = dict(y=y[:inf_bs, ...], precision_dtype=precision_dtype, enable_amp=amp)
        if input_cfg >= 0:
            model_kwargs['cfg_scale'] = input_cfg
        for samples in sample_fn(fwd_fn, inf_x.shape, inf_x, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device):
            final_samples = samples
        recon_param = final_samples["sample"].reshape(inf_bs, config.model.num_prims, -1)
        if perchannel_norm:
            recon_param = recon_param / latent_nf * latent_std + latent_mean
        recon_srt_param = recon_param[:, :, 0:4]
        recon_feat_param = recon_param[:, :, 4:] # [8, 2048, 64]
        recon_feat_param_list = []
        # one-by-one to avoid oom
        for inf_bidx in range(inf_bs):
            if not perchannel_norm:
                decoded = vae.decode(recon_feat_param[inf_bidx, ...].reshape(1*config.model.num_prims, *latent.shape[-4:]) / latent_nf)
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
        visualize_video_primvolume(config.output_dir, batch, recon_param, 15, rm, device)
        prim_params = {'srt_param': recon_srt_param[0].detach().cpu(), 'feat_param': recon_feat_param[0].detach().cpu()}
        torch.save({'model_state_dict': prim_params}, "{}/denoised.pt".format(config.output_dir))

    return output_rgb_video_path, output_prim_video_path, output_mat_video_path, gr.update(interactive=True)

def export_mesh(remesh="No", mc_resolution=256, decimate=100000):
    # exporting GLB mesh
    output_glb_path = os.path.join(config.output_dir, GRADIO_GLB_PATH)
    if remesh == "No":
        config.inference.remesh = False
    elif remesh == "Yes":
        config.inference.remesh = True
    config.inference.decimate = decimate
    config.inference.mc_resolution = mc_resolution
    config.inference.batch_size = 8192
    denoise_param_path = os.path.join(config.output_dir, 'denoised.pt')
    primx_ckpt_weight = torch.load(denoise_param_path, map_location='cpu')['model_state_dict']
    model_primx.load_state_dict(primx_ckpt_weight)
    model_primx.to(device)
    model_primx.eval()
    with torch.no_grad():
        model_primx.srt_param[:, 1:4] *= 0.85
        extract_texmesh(config.inference, model_primx, config.output_dir, device)
    return output_glb_path, gr.update(visible=True)

# gradio UI
_TITLE = '''3DTopia-XL'''

_DESCRIPTION = '''
<div>
<a style="display:inline-block" href="https://frozenburning.github.io/projects/3DTopia-XL/"><img src='https://img.shields.io/badge/public_website-8A2BE2'></a>
<a style="display:inline-block; margin-left: .5em" href="https://github.com/3DTopia/3DTopia-XL"><img src='https://img.shields.io/github/stars/3DTopia/3DTopia-XL?style=social'/></a>
</div>

* Now we offer 1) single image conditioned model, we will release 2) multiview images conditioned model and 3) pure text conditioned model in the future!
* If you find the output unsatisfying, try using different seeds!
'''

block = gr.Blocks(title=_TITLE).queue()
with block:
    current_fg_state = gr.State()
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('# ' + _TITLE)
    gr.Markdown(_DESCRIPTION)
    
    with gr.Row(variant='panel'):
        with gr.Column(scale=1):
            with gr.Row():
                # input image
                input_image = gr.Image(label="image", type='pil')
                # background removal
                removal_previewer = gr.Image(label="Background Removal Preview", type='pil', interactive=False)
            # inference steps
            input_num_steps = gr.Radio(choices=[25, 50, 100, 200], label="DDIM steps", value=25)
            # random seed
            input_cfg = gr.Slider(label="CFG scale", minimum=0, maximum=15, step=0.5, value=6, info="Typically CFG in a range of 4-7")
            # random seed
            input_seed = gr.Slider(label="random seed", minimum=0, maximum=10000, step=1, value=42, info="Try different seed if the result is not satisfying as this is a generative model!")
            # gen button
            button_gen = gr.Button(value="Generate", interactive=False)
            with gr.Row():
                input_mc_resolution = gr.Radio(choices=[64, 128, 256], label="MC Resolution", value=128, info="Cube resolution for mesh extraction")
                input_remesh = gr.Radio(choices=["No", "Yes"], label="Remesh", value="No", info="Remesh or not?")
            export_glb_btn = gr.Button(value="Export GLB", interactive=False)

        with gr.Column(scale=1):
            with gr.Row():
                # final video results
                output_rgb_video = gr.Video(label="RGB")
                output_prim_video = gr.Video(label="Primitives")
                output_mat_video = gr.Video(label="Material")
            with gr.Row():
                # glb file
                output_glb = LitModel3D(
                    label="3D GLB Model",
                    visible=True,
                    clear_color=[0.0, 0.0, 0.0, 0.0],
                    camera_position=(90, None, None),
                    tonemapping="aces",
                    contrast=1.0,
                    scale=1.0,
                )
            with gr.Column(visible=False, scale=1.0) as hdr_row:
                gr.Markdown("""## HDR Environment Map
                
                Select / Upload an HDR environment map to relight the 3D model.
                """)
                with gr.Row():
                    example_hdris = [
                        os.path.join("assets/hdri", f)
                        for f in os.listdir("assets/hdri")
                    ]
                    hdr_illumination_file = gr.File(
                        label="HDR Envmap", file_types=[".hdr"], file_count="single"
                    )
                    hdr_illumination_example = gr.Examples(
                        examples=example_hdris,
                        inputs=hdr_illumination_file,
                    )

                    hdr_illumination_file.change(
                        lambda x: gr.update(env_map=x.name if x is not None else None),
                        inputs=hdr_illumination_file,
                        outputs=[output_glb],
                    )

    input_image.change(background_remove_process, inputs=[input_image], outputs=[button_gen, current_fg_state, removal_previewer])
    button_gen.click(process, inputs=[current_fg_state, input_num_steps, input_seed, input_cfg], outputs=[output_rgb_video, output_prim_video, output_mat_video, export_glb_btn])
    export_glb_btn.click(export_mesh, inputs=[input_remesh, input_mc_resolution], outputs=[output_glb, hdr_row])
    
    gr.Examples(
        examples=[
            os.path.join("assets/examples", f)
            for f in os.listdir("assets/examples")
        ],
        inputs=[input_image],
        outputs=[output_rgb_video, output_prim_video, output_mat_video, export_glb_btn],
        fn=lambda x: process(input_image=x),
        cache_examples=False,
        label='Single Image to 3D PBR Asset'
    )
    
block.launch(server_name="0.0.0.0", share=True)