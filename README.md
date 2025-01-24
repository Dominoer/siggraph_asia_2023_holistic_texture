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

## Dataset Preparation

To prepare your training dataset:

1. Collect texture images from open-source datasets or web sources

2. Filter the images based on these criteria:
   - Remove images with occlusions
   - Remove images with distortions
   - Remove images with perspective variations

3. Organize your dataset with the following structure:
```
dataset/
├── train/
│   └── *.jpg
├── valid/
│   └── *.jpg
└── test/
    └── *.jpg
```

## VQ-VAE checkpoint

We provide a VQ-VAE model that has been fine-tuned on our texture dataset to improve reconstruction performance for texture images.
1. Download our fine-tuned VQ-VAE checkpoint from [[Google Drive](https://drive.google.com/file/d/1s6L0fFDuYyS2aXHCwgLSSd10a3Ms1GiM/view?usp=sharing)]
2. Place the checkpoint in the following directory:
```
'./logs/autoencoder_vq_32x32xx4/checkpoints'
```

## Training

Specify paths in the config file before runing the script.
```
python main.py --base configs/latent-diffusion/texture-ldm-vq-8.yaml -t --gpus 0
```

## Inference

1. Download pretrained model from [[Google Drive](https://drive.google.com/file/d/177p45j30pN9FlU1h3NFEz6l75Zo0dU5a/view?usp=drive_link)]

2. Place the checkpoint in the following directory:
```
'./logs/texture-ldm-vq-8/checkpoints'
```

3. Prepare a real-world image containing the desired texture, along with a corresponding mask indicating the target region. Then run:
```
python inference.py --image /path_to_image/*.jpg --mask /path_to_mask/*.png
```

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