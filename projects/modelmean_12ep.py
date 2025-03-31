import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--folder', type=str, help='folder path')
args = parser.parse_args()

import torch
x0 = torch.load(f'{args.folder}/model_0089999.pth', map_location='cpu')
x2 = torch.load(f'{args.folder}/model_0029999.pth', map_location='cpu')
x4 = torch.load(f'{args.folder}/model_0059999.pth', map_location='cpu')

new_dict = {}
for k, v in x0['model'].items():
    new_dict[k] = x0['model'][k] * 0.8 + x2['model'][k] * 0.15 + x4['model'][k] * 0.05
torch.save({"model": new_dict}, f"{args.folder}/meanmodel.pth", )