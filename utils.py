import argparse


import torch
x0 = torch.load(f'output/mask_ours_24ep.pth', map_location='cpu')


new_dict = {}
for k, v in x0['model'].items():
    new_dict[k] = x0['model'][k] 
torch.save({"model": new_dict}, f"output/MrDETR_deformable_r50_24ep_300q_insseg.pth", )