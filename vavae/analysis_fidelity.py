import argparse
import os
import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import torch_fidelity
import cv2

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import Image
from pytorch_wavelets import DWTForward, DWTInverse
from tqdm import tqdm
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser('VAE Analysis', add_help=False)
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--data_path', default='/BS/var/nobackup/imagenet-1k/', type=str)
    parser.add_argument('--resos', default=256, type=int)
    args = parser.parse_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # data loading
    transform = transforms.Compose([
        transforms.Resize(args.resos),
        transforms.CenterCrop((args.resos, args.resos)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = ImageFolder(root=os.path.join(args.data_path, 'val'), transform=transform)
    image_val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # build the model
    config_high = OmegaConf.load('configs/f16d32_vfdinov2_high.yaml')
    vae_high = instantiate_from_config(config_high.model)
    vae_high.init_from_ckpt(config_high.ckpt_path)
    vae_high = vae_high.to(device)
    vae_high.eval()
    for p in vae_high.parameters(): p.requires_grad_(False)

    config_low = OmegaConf.load('configs/f16d32_vfdinov2_low.yaml')
    vae_low = instantiate_from_config(config_low.model)
    vae_low.init_from_ckpt(config_low.ckpt_path)
    vae_low = vae_low.to(device)
    vae_low.eval()
    for p in vae_low.parameters(): p.requires_grad_(False)
    print(f'prepare finished.')

    dwt = DWTForward(J=1, wave='haar', mode='zero').to(device)
    idwt = DWTInverse('haar', mode='zero').to(device)

    ref_dir = '/BS/var/nobackup/recon/ground_truth'
    save_dir = '/BS/var/nobackup/recon/va-vae-high-low-f16c32'
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # save reference images if needed
    ref_png_files = [f for f in os.listdir(ref_dir) if f.endswith('.png')]
    save_ref = bool(len(ref_png_files) < 50000)

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(image_val_loader, desc='Processing images', leave=True)):
            imgs = imgs.to(device)
            # first level DWT
            ll1, hs = dwt(imgs)
            
            # low frequency components
            low = ll1 / 2 # normalize
            low = torch.nn.functional.interpolate(low, size=256, mode='bicubic', align_corners=False)
            posterior_low = vae_low.encode(low)
            z_low = posterior_low.sample()
            rec_low = vae_low.decode(z_low)
            rec_low = torch.nn.functional.interpolate(rec_low, size=128, mode='bicubic', align_corners=False)
            rec_low = rec_low * 2 # denormalize
            
            # high frequency components
            high = hs[0] / 2 # normalize
            high = high.view(-1, 9, 128, 128)
            posterior_high = vae_high.encode(high)
            z_high = posterior_high.sample()
            rec_high = vae_high.decode(z_high)
            rec_high = rec_high.view(-1, 3, 3, 128, 128)
            rec_high = rec_high * 2 # denormalize
            
            # reconstruct images from wavelet components
            rec_imgs = idwt((rec_low, [rec_high]))
            
            if save_ref:
                imgs = torch.clamp((imgs + 1) / 2, min=0, max=1)
                for b_id in range(imgs.size(0)):
                    img_id = i * imgs.size(0) + b_id
                    img = np.round(imgs[b_id].cpu().numpy().transpose([1, 2, 0]) * 255)
                    img = img.astype(np.uint8)[:, :, ::-1]
                    cv2.imwrite(os.path.join(ref_dir, f'{img_id:05}.png'), img)

            rec_imgs = torch.clamp((rec_imgs + 1) / 2, min=0, max=1)
            for b_id in range(rec_imgs.size(0)):
                img_id = i * rec_imgs.size(0) + b_id
                img = np.round(rec_imgs[b_id].cpu().numpy().transpose([1, 2, 0]) * 255)
                img = img.astype(np.uint8)[:, :, ::-1]
                cv2.imwrite(os.path.join(save_dir, f'{img_id:05}.png'), img)

    metrics_dict = torch_fidelity.calculate_metrics(
        input1=save_dir,
        input2=ref_dir,
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=True,
    )
    fid = metrics_dict['frechet_inception_distance']
    inception_score = metrics_dict['inception_score_mean']
    print(f'{fid=}, {inception_score=}')
