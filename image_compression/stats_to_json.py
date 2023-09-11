import json
import glob
import os

import yaml
from absl import app
from absl import flags

import dataio

flags.DEFINE_string('exp_folder', None, None)
flags.DEFINE_integer('bitwidth', 8, None)
flags.DEFINE_string('out_folder', None, None)
FLAGS = flags.FLAGS

def calculate_metrics(yaml_stats):
    return {
        "psnr": yaml_stats["psnr"],
        "ssim": yaml_stats["ssim"],
        "state_bpp": yaml_stats["bpp"],
        "bpp": yaml_stats["bpp"],
    }

def main(_):
    yaml_stats_glob = f"*metrics_arithmetic_bw{FLAGS.bitwidth}*"
    yaml_stats_path = glob.glob(os.path.join(FLAGS.exp_folder, yaml_stats_glob))[0]
    yaml_stats = yaml.safe_load(open(yaml_stats_path, "r"))

    stats = calculate_metrics(yaml_stats)

    print(stats)
    json.dump(stats, open(os.path.join(FLAGS.out_folder, "stats.json"), "w"))

if __name__ == '__main__':
    app.run(main)
