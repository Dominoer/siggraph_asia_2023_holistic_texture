import math
import random
import argparse, os, sys
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
import numpy as np
import torch

from torchvision import transforms as transforms

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def read_batch(image_path, mask_path, device):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("RGB")
    mask = np.asarray(mask, np.uint8).transpose(2, 0, 1)
    mask = np.where(mask > 180, 255, 0).astype(np.uint8)
    mask = torch.from_numpy(mask).float() / 255.

    image = transforms.ToTensor()(image)
    image = transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))(image)
    image = transforms.Resize((256, 256),
                              transforms.InterpolationMode.BILINEAR)(image)

    masked = image * mask
    # Create binary mask: 1's where we have content, 0's where we do not.
    mask = torch.where(masked == 0,
                       torch.zeros_like(masked),
                       torch.ones_like(masked))

    with_mask = torch.cat([masked, mask], dim=0)
    
    return (dict(c_concat=masked.unsqueeze(0).to(device),
                 c_crossattn=with_mask.unsqueeze(0).to(device)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the single input image (e.g. 'example_img.jpg')."
    )
    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        help="Path to the single input mask (e.g. 'example_mask.tif')."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of DDIM sampling steps."
    )

    opt = parser.parse_args()

    # Load config and model
    # config = OmegaConf.load("logs/texture-ldm-vq-8/configs/2024-08-22T15-50-51-project.yaml")
    config = OmegaConf.load("configs/latent-diffusion/texture-ldm-vq-8.yaml")
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load("logs/texture-ldm-vq-8/checkpoints/last.ckpt")["state_dict"],
                          strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()
    sampler = DDIMSampler(model)

    # Prepare paths
    base_name = os.path.basename(opt.image)
    # outpath = os.path.join(opt.outdir, base_name)
    outpath = os.path.join('./', 'rectified_' + base_name)

    # Read image and mask, construct the conditioning
    conds = read_batch(opt.image, opt.mask, device=device)

    with torch.no_grad():
        with model.ema_scope():
            # Inference
            vis_concat = conds["c_concat"].clone()
            conds["c_concat"] = [model.encode_first_stage(conds["c_concat"]).detach()]
            conds["c_crossattn"] = [conds["c_crossattn"]]

            shape = (4, 32, 32)
            samples_ddpm, _ = sampler.sample(
                S=opt.steps,
                batch_size=1,
                conditioning=conds,
                shape=shape,
                verbose=False
            )
            x_samples_ddpm = model.decode_first_stage(samples_ddpm)

            # Convert output to numpy image
            rectified = torch.clamp((x_samples_ddpm+1.0)/2.0, min=0.0, max=1.0).squeeze()
            rectified = rectified.cpu().numpy()
            rectified = np.transpose(rectified * 255, (1, 2, 0))
            Image.fromarray(rectified.astype(np.uint8)).save(outpath)
    
    print(f"Done. Rectified image saved to: {outpath}")


# /mnt/hdd/datasets/texture/Commercial_use/roi/11644774_abff070e97_w.jpg
# /mnt/hdd/datasets/texture/Commercial_use/mask/11644774_abff070e97_w.tif