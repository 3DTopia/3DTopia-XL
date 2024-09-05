<div align="center">

<h1>3DTopia-XL: High-Quality 3D PBR Asset Generation via Primitive Diffusion</h1>

<div>

<a target="_blank" href="https://arxiv.org/abs/xxxx.xxxxx">
  <img src="https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg" alt="arXiv Paper"/>
</a>
<a target="_blank" href="https://huggingface.co/spaces/FrozenBurning/3DTopia-XL">
  <img src="https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue" alt="HuggingFace"/>
</a>
<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2F3DTopia%2F3DTopia-XL&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
</div>


<h4>TL;DR</h4>
<h5>3DTopia-XL is a 3D diffusion transformer (DiT) operating on primitive-based representation. <br>
It can generate 3D asset with smooth geometry and PBR materials from single image or text.</h5>

### [Paper](https://arxiv.org/abs/xxxx.xxxxx) | [Project Page](https://frozenburning.github.io/projects/3DTopia-XL/) | [Weights](https://huggingface.co/FrozenBurning/3DTopia-XL) | [Hugging Face :hugs:](https://huggingface.co/spaces/FrozenBurning/3DTopia-XL)

<br>

<video controls autoplay src="https://github.com/user-attachments/assets/6e281d2e-9741-4f81-ae57-c4ce30b36356"></video>

</div>

## News

[09/2024] Hugging Face demo released! [![demo](https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/FrozenBurning/3DTopia-XL)

[08/2024] Inference code released!

## Installation
We highly recommend using [Anaconda](https://www.anaconda.com/) to manage your python environment. You can setup the required environment by the following commands:
```bash
# install dependencies
conda create -n primx python=3.9
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
# requires xformer for efficient attention
conda install xformers::xformers
# install other dependencies
pip install -r requirements.txt
# compile third party libraries
bash install.sh
# Now, all done!
```

## Pretrained Weights

Our pretrained weight can be downloaded from [huggingface](https://huggingface.co/FrozenBurning/3DTopia-XL)

For example, to download the singleview-conditioned model in fp16 precision for inference:
```bash
mkdir pretrained && cd pretrained
# download DiT
wget https://huggingface.co/FrozenBurning/3DTopia-XL/resolve/main/model_sview_dit_fp16.pt
# download VAE
wget https://huggingface.co/FrozenBurning/3DTopia-XL/resolve/main/model_vae_fp16.pt
cd ..
```

We will release the multiview-conditioned model and text-conditioned model in the near future!

## Inference

### Gradio Demo
The gradio demo will automatically download pretrained weights using huggingface_hub.

You could locally launch our demo with Gradio UI by:
```bash
python app.py
```
Alternatively, you can run the demo online [![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/FrozenBurning/3DTopia-XL)

### CLI Test
Run the following command for inference:
```bash
python inference.py ./configs/inference_dit.yml
```
Furthermore, you can modify the inference parameters in [inference_dit.yml](./configs/inference_dit.yml), detailed as follows:

| Parameter | Recommended | Description |
| :---------- | :------------: | :---------- |
| `input_dir` | - | The path of folder that stores all input images. |
| `ddim` | 25, 50, 100 | Total number of DDIM steps. Robust with more steps but fast with fewer steps. |
| `cfg` | 4 - 7 | The scale for Classifer-free Guidance (CFG). |
| `seed` | Any | Different seeds lead to diverse different results.|
| `export_glb` | True | Whether to export textured mesh in GLB format after DDIM sampling is over. |
| `fast_unwrap` | False | Whether to enable fast UV unwrapping algorithm. |
| `decimate` | 100000 | The max number of faces for mesh extraction. |
| `mc_resolution` | 256 | The resolution of the unit cube for marching cube. |
| `remesh` | False | Whether to run retopology after mesh extraction. |



## Training

We will release the training code and details in the future!

## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks all the authors for sharing!

- [PrimDiffusion](https://github.com/FrozenBurning/PrimDiffusion)
- [MVP](https://github.com/facebookresearch/mvp)
- [DiT](https://github.com/facebookresearch/DiT)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
- [kiuikit](https://github.com/ashawkey/kiuikit)
- [Trimesh](https://github.com/mikedh/trimesh)
- [litmodel3d](https://pypi.org/project/gradio-litmodel3d/)
