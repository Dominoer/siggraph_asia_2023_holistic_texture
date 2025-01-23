# Holistic Texture Rectification and Synthesis
PyTorch implementation for "[Diffusion-based Holistic Texture Rectification and Synthesis](https://arxiv.org/pdf/2309.14759)" at SIGGRAPH Asia 2023.


[Project Page](https://dominoer.github.io/siggraph_asia_2023_holistic_texture/)

## Installation
1. Clone this repository:
```
git clone https://github.com/Dominoer/siggraph_asia_2023_holistic_texture.git
cd siggraph_asia_2023_holistic_texture
```

2. Create and activate the conda environment:
```
conda env create -f env.yml
conda activate ldm
```
This environment is built upon the Latent Diffusion Model's dependencies with Kornia library. 


## VQ-VAE checkpoint
We provide a VQ-VAE model that has been fine-tuned on our texture dataset to improve reconstruction performance for texture images.
1. Download our fine-tuned VQ-VAE checkpoint from [[Google Drive/URL](https://drive.google.com/file/d/1s6L0fFDuYyS2aXHCwgLSSd10a3Ms1GiM/view?usp=sharing)]
2. Place the checkpoint in the following directory:
```
'./logs/autoencoder_vq_32x32xx4/checkpoints'
```

## Training
Specify paths in the train.sh script before runing the script.
```
python main.py --base configs/latent-diffusion/texture-ldm-vq-8.yaml -t --gpus 0
```

## ToDo
 - [ ] Online demo on HuggingFace

## Citation
If you find the code useful in your research, please consider citing our paper:
```
@Inproceedings{HaoSIGGRAPHASIA2023,
		author    = {Guoqing Hao and Satoshi Iizuka and Kensho Hara and Edgar Simo-Serra and Hirokatsu Kataoka and Kazuhiro Fukui},
		title     = {Diffusion-based Holistic Texture Rectification and Synthesis},
		booktitle = "ACM SIGGRAPH Asia 2023 Conference Papers",
		year      = 2023,
	 }
```