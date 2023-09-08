import json
import glob
import os
from skimage import io

from pytorch_msssim import ms_ssim, ssim

import torch
import yaml
from absl import app
from absl import flags
from torch.utils.data import DataLoader

import dataio
import modules

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

flags.DEFINE_string('image_path', None, None)
flags.DEFINE_string('flags_file', None, None)
flags.DEFINE_string('exp_folder', None, None)
flags.DEFINE_integer('bitwidth', 8, None)
flags.DEFINE_string('out_folder', None, None)
FLAGS = flags.FLAGS

def infer(test_loader: DataLoader, model: torch.nn.Module, image_resolution):
    model.eval()
    with torch.no_grad():
        for step, (model_input, gt) in enumerate(test_loader):
            model_input = {key: value.cuda() for key, value in model_input.items()}
            gt = {key: value.cuda() for key, value in gt.items()}

            predictions = model(model_input)
            original = dataio.lin2img(gt['img'], image_resolution)
            reconstructed = dataio.lin2img(predictions['model_out'], image_resolution)
            original = original.squeeze()
            reconstructed = reconstructed.squeeze()
            original = original.div(2.0).add(0.5).clamp(0.0, 1.0).mul(255.0)
            reconstructed = reconstructed.div(2.0).add(0.5).clamp(0.0, 1.0).mul(255.0)
            return original, reconstructed

def psnr(img1, img2, range=255.0):
    mse = torch.mean((img1 - img2) ** 2)
    psnr = 20 * torch.log10(range / torch.sqrt(mse))

    return psnr

def ms_ssim_reshape(tensor):
    return tensor.unsqueeze(0)

def calculate_metrics(original, reconstructed, yaml_stats):
    pixels = original.nelement() / 3.0

    return {
        "psnr": psnr(original, reconstructed).item(),
        "ssim": ssim(ms_ssim_reshape(original), ms_ssim_reshape(reconstructed)).item(),
        "ms-ssim": ms_ssim(ms_ssim_reshape(original), ms_ssim_reshape(reconstructed)).item(),
        "state_bpp": yaml_stats["bpp"],
        "bpp": yaml_stats["bpp"],
    }

def main(_):
    TRAINING_FLAGS = yaml.safe_load(open(FLAGS.flags_file, 'r'))

    img = dataio.ImageFile(FLAGS.image_path)
    image_resolution = (img.img.size[1], img.img.size[0])

    model = modules.INRNet(type=TRAINING_FLAGS['activation'], mode=TRAINING_FLAGS['encoding'],
                            sidelength=image_resolution,
                            out_features=img.img_channels,
                            hidden_features=TRAINING_FLAGS['hidden_dims'],
                            num_hidden_layers=TRAINING_FLAGS['hidden_layers'],
                            encoding_scale=TRAINING_FLAGS['encoding_scale'],
                            ff_dims=TRAINING_FLAGS['ff_dims'])

    coord_dataset = dataio.Implicit2DWrapper(img, sidelength=image_resolution)

    dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=1, pin_memory=True,
                            num_workers=0)

    model_glob = f"*model_bw{FLAGS.bitwidth}*"
    state_path = glob.glob(os.path.join(FLAGS.exp_folder, model_glob))[0]
    state_dict = torch.load(state_path)
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)

    original, reconstructed = infer(dataloader, model, image_resolution)

    yaml_stats_glob = f"*metrics_arithmetic_bw{FLAGS.bitwidth}*"
    yaml_stats_path = glob.glob(os.path.join(FLAGS.exp_folder, yaml_stats_glob))[0]
    yaml_stats = yaml.safe_load(open(yaml_stats_path, "r"))

    stats = calculate_metrics(original, reconstructed, yaml_stats)

    print(stats)
    json.dump(stats, open(os.path.join(FLAGS.out_folder, "stats.json"), "w"))

    io.imsave(os.path.join(FLAGS.out_folder, "decoded.png"), 
              reconstructed.to(torch.uint8).permute(1, 2, 0).cpu().numpy())

if __name__ == '__main__':
    app.run(main)
