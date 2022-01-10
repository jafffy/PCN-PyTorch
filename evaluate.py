import argparse
import torch
import os
from dataset.dataset import S3DIS
from model import AutoEncoder
from loss import ChamferDistance


parser = argparse.ArgumentParser()
parser.add_argument('--complete_path', type=str, default="/home/ubuntu/Projects/PCD_completion/dataset/S3DIS/processed/area1_pruned4_complete.pcv")
parser.add_argument('--incomplete_path', type=str, default="/home/ubuntu/Projects/PCD_completion/dataset/S3DIS/processed/area1_pruned4_incomplete_seed_10_0.2.pcv")
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

cd_loss = ChamferDistance()

test_dataset = S3DIS(incomplete_path=args.incomplete_path, complete_path=args.complete_path, split='test')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

network = AutoEncoder()
network.load_state_dict(torch.load('log/pretrained_model.pth'))
network.to(DEVICE)

# testing: evaluate the mean cd loss
network.eval()
with torch.no_grad():
    total_loss, iter_count = 0, 0
    for i, data in enumerate(test_dataloader, 1):
        incomplete_input, complete_input = data

        incomplete_input = incomplete_input.to(DEVICE)
        complete_input = complete_input.to(DEVICE)
        incomplete_input = incomplete_input.permute(0, 2, 1)
        
        v, y_coarse, y_detail = network(incomplete_input)

        y_detail = y_detail.permute(0, 2, 1)

        loss = cd_loss(complete_input, y_detail)
        total_loss += loss.item()
        iter_count += 1

    mean_loss = total_loss / iter_count
    print("\033[31mTesting loss is {}\033[0m".format(mean_loss))
