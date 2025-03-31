import argparse


import torch
x0 = torch.load(f'output/MrDETR_deformable_swinl_12ep_900q_objects365.pth', map_location='cpu')


new_dict = {}
for k, v in x0['model'].items():
    new_dict[k] = x0['model'][k] 
torch.save({"model": new_dict}, f"output/MrDETR_deformable_swinl_12ep_900q_objects365_processed.pth", )