import argparse

import numpy as np
import torch
import os
from dataset.dataset import S3DIS
from model import AutoEncoder
import sys
sys.path.append('./metasdf-for-pcd-completion/PCN-PyTorch/')
from loss import ChamferDistance
from datetime import datetime
sys.path.append('../..')
import config
from dataset.utils import PCDUtil
import os

cur_dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="/home/ubuntu/Projects/PCD_completion/dataset/S3DIS/processed")

parser.add_argument('--complete_path', type=str, default="area1_pruned4_complete.pcv")
parser.add_argument('--incomplete_path', type=str, default="area1_pruned4_incomplete_seed_10_0.2.pcv")
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

cd_loss = ChamferDistance()
args.incomplete_path = os.path.join(args.dir, args.incomplete_path)
args.complete_path = os.path.join(args.dir, args.complete_path)
test_dataset = S3DIS(incomplete_path=args.incomplete_path, complete_path=args.complete_path, split='test')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

network = AutoEncoder()

network.load_state_dict(torch.load(os.path.join(cur_dir, 'log/pretrained_model.pth')))
network.to(DEVICE)

# testing: evaluate the mean cd loss
network.eval()

current_time = datetime.now().strftime("%y%m%d%H%M%S")
result_folder = os.path.join(config.RESULTS_DIR, f"PCN_{current_time}")
os.mkdir(result_folder)
with torch.no_grad():
    total_loss, iter_count = 0, 0
    total_bloss = 0
    for i, data in enumerate(test_dataloader, 1):
        incomplete_input, complete_input, rgb = data

        incomplete_input = incomplete_input.to(DEVICE)
        complete_input = complete_input.to(DEVICE)
        incomplete_input = incomplete_input.permute(0, 2, 1)
        
        v, y_coarse, y_detail = network(incomplete_input)

        y_detail = y_detail.permute(0, 2, 1)

        total_bloss += cd_loss(complete_input, incomplete_input)

        loss = cd_loss(complete_input, y_detail)
        total_loss += loss.item()
        iter_count += 1

        y_output = y_detail.detach().cpu().numpy()[0]
        incomplete_input = incomplete_input.detach().cpu().numpy()[0].transpose(1,0)
        geos = np.concatenate([incomplete_input, y_output])
        print(incomplete_input.shape, y_output.shape, geos.shape)
        y_rgbs = np.array([[255,0,255] for _ in range(y_output.shape[0])])
        rgb = rgb.numpy()[0]
        rgbs = np.concatenate([rgb, y_rgbs])
        print(rgb.shape, y_rgbs.shape, rgbs.shape)

        # dataset.utils.PCDUtil.write_pcvfile(rgbs, geos, os.path.join(result_folder, "detail.pcv"))

        PCDUtil.write_pcd_from_geo_rgb(rgbs, geos, os.path.join(result_folder, "detail.ply"))

    mean_loss = total_loss / iter_count
    print("\033[31mTesting loss is {}, Base loss is {}\033[0m".format(mean_loss, total_bloss/iter_count))
