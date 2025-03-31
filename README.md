# Mr. DETR
**<center><font size=4>[CVPR 2025] Mr. DETR: Instructive Multi-Route Training for Detection Transformers</font></center>**  
[Chang-Bin Zhang](https://zhangchbin.github.io)<sup>1</sup>, Yujie Zhong<sup>2</sup>, Kai Han<sup>1</sup>  
<sup>1</sup> <sub>The University of Hong Kong</sub>  
<sup>2</sup> <sub>Meituan Inc.</sub>  

[![Conference](https://img.shields.io/badge/CVPR-2025-blue)]()
[![Paper](https://img.shields.io/badge/arXiv-2412.10028-brightgreen)](https://arxiv.org/abs/2412.10028)
[![Project](https://img.shields.io/badge/Project-red)](https://visual-ai.github.io/mrdetr/)
<a href="mailto: zhangchbin@gmail.com">
        <img alt="emal" src="https://img.shields.io/badge/contact_me-email-yellow">
    </a>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mr-detr-instructive-multi-route-training-for/object-detection-on-coco-2017-val)](https://paperswithcode.com/sota/object-detection-on-coco-2017-val?p=mr-detr-instructive-multi-route-training-for)


## Updates
- [03/25] We release the code and weights of Mr. DETR for object detection.
- [03/25] Mr. DETR is accepted by CVPR 2025.

## Performance
[Demo Video for Street](https://www.bilibili.com/video/BV1ThZnYxE5G/?spm_id_from=333.1387.homepage.video_card.click&vd_source=3b32a049a039d0ef814f8588b3c9b2d9)  
[Demo Video for Dense and Crowded Scene](https://www.zhihu.com/zvideo/1890060966391153546)


## Method
<img width="1230" alt="" src="assets/mrdetrmethod.png">


## Model Zoo
| Model | Backbone | Query | Epochs | AP | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>s</sub> | AP<sub>m</sub> | AP<sub>l</sub> |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Mr. DETR-Deformable | R50 | 300 | 12 | 49.5 | 67.0 | 53.7 | 32.1 | 52.5 | 64.7 | [Config](projects/mr_detr_deformable/configs/deformable_detr_r50_two_stage_12ep_plusplus_300q.py) & [Weights](https://huggingface.co/allencbzhang/Mr.DETR/resolve/main/MrDETR_deformable_r50_12ep_300q.pth) |
| Mr. DETR-Deformable | R50 | 900 | 12 | 50.7 | 68.2 | 55.4 | 33.6 | 54.3 | 64.6 | [Config](projects/mr_detr_deformable/configs/deformable_detr_r50_two_stage_12ep_plusplus.py) & [Weights](https://huggingface.co/allencbzhang/Mr.DETR/resolve/main/MrDETR_deformbale_r50_12ep_900q.pth) |
| Mr. DETR-Deformable | R50 | 900 | 24 | 51.4 | 69.0 | 56.2 | 34.9 | 54.8 | 66.0 | [Config](projects/mr_detr_deformable/configs/deformable_detr_r50_two_stage_24ep_plusplus.py) & [Weights](https://huggingface.co/allencbzhang/Mr.DETR/resolve/main/MrDETR_deformable_r50_24ep_900q.pth) |
| Mr. DETR-DINO | R50 | 900 | 12 | 50.9 | 68.4 | 55.6 | 34.6 | 53.8 | 65.2 | [Config](projects/mr_detr_dino/configs/deformable_detr_r50_two_stage_12ep_plusplus.py) & [Weights](https://huggingface.co/allencbzhang/Mr.DETR/resolve/main/MrDETR_dino_r50_12ep_900q.pth) |
| Mr. DETR-Align | R50 | 900 | 12 | 51.4 | 68.6 | 55.7 | 33.8 | 54.7 | 66.3 | [Config](projects/mr_detr_align/configs/deformable_detr_r50_two_stage_12ep_plusplus.py) & [Weights](https://huggingface.co/allencbzhang/Mr.DETR/resolve/main/MrDETR_align_r50_12ep_900q.pth) |
| Mr. DETR-Align | R50 | 900 | 24 | 52.3 | 69.5 | 56.7 | 35.2 | 56.0 | 67.0 | [Config](projects/mr_detr_align/configs/deformable_detr_r50_two_stage_24ep_plusplus.py) & [Weights](https://huggingface.co/allencbzhang/Mr.DETR/resolve/main/MrDETR_align_r50_24ep_900q.pth) |
| Mr. DETR-Align | Swin-L | 900 | 12 | 58.4 | 76.3 | 63.9 | 40.8 | 62.8 | 75.3 | [Config](projects/mr_detr_align/configs/deformable_detr_swinl_two_stage_12ep_plusplus.py) & [Weights](https://huggingface.co/allencbzhang/Mr.DETR/resolve/main/MrDETR_align_swinL_12ep_900q.pth) |
| Mr. DETR-Align<sup>*</sup> | Swin-L | 900 | 12 | 61.8 | 79.0 | 67.6 | 47.7 | 65.6 | 75.7 | [Config](projects/mr_detr_align/configs/deformable_detr_swinl_two_stage_12ep_plusplus_5scale_fintuning.py) & [Weights](https://huggingface.co/allencbzhang/Mr.DETR/resolve/main/MrDETR_align_swinl_12ep_900q_objects365_processed.pth) |

***: The model is fine-tuned on the Objects365 Pretrained Model with 5-scale. Due to the limited GPU resources, we only pre-trained the Swin-L based Mr. DETR for 549K iterations (batchsize of 16).**



## Environment Setup
- This repository is based on the Detrex framework, thus you may refer to [installation docs](https://detrex.readthedocs.io/en/latest/tutorials/Installation.html).
- Python $\ge$ 3.7 and PyTorch $\ge$ 1.10 are required.  
- First, clone ```Mr. DETR``` repository and initialize the ```detectron2``` submodule.
```
git clone https://github.com/Visual-AI/Mr.DETR.git
cd Mr.DETR
git submodule init
git submodule update
```
- Second, install ```detectron2``` and ```detrex```
```
pip install -e detectron2
pip install -r requirements.txt
pip install -e .
```

- If you encounter any ```compilation error of cuda runtime```, you may try to use
```
export CUDA_HOME=<your_cuda_path>
```

- You may start with COCO 2017 dataset, which is organized as:
```
datasets/
└── coco2017/
    │
    ├── annotations/                  
    │   ├── instances_train2017.json  
    │   └── instances_val2017.json    
    │
    ├── train2017/                    
    │   └── ...
    │
    └── val2017/                   
        └── ...
```
- Then set the path of ```DETECTRON2_DATASETS``` by
```
export DETECTRON2_DATASETS=<.../datasets/>
```

## API and Demo
You may also refer to the [document](https://detrex.readthedocs.io/en/latest/tutorials/Getting_Started.html).
- Visualize an image:
```
python demo/demo.py --config-file <config_file> \
                    --input assets/000000028449.jpg \
                    --output visualized_000000028449.jpg \
                    --confidence-threshold 0.5 \
                    --opts train.init_checkpoint=<checkpoint_path> 
```

- Visualize a video:
```
python demo/demo.py --config-file <config_file> \
                    --video-input xxx.mp4 \
                    --output visualized.mp4 \
                    --confidence-threshold 0.5 \
                    --opts train.init_checkpoint=<checkpoint_path> 
```

- Visualize test results:
```
python tools/visualize_json_results.py --input /path/to/x.json \ # path to the saved testing results
                                       --output dir/ \
                                       --dataset coco_2017_val
```




## Train
- For R50 based models:
```
python projects/train_net.py \
    --config-file <config-file> \
    --num-gpus N \
    dataloader.train.total_batch_size=16 \
    train.output_dir=<output_dir> \
    train.amp.enabled=True \ # mixed precision training
    model.transformer.encoder.use_checkpoint=True \ # gradient checkpointing, save gpu memory but lower speed

# to get mean model, which is more stable than ema, and improves about 0.1~0.2%.
python projects/modelmean_12ep.py --folder <output_dir>
python projects/modelmean_24ep.py --folder <output_dir>

python projects/train_net.py \
    --config-file <config-file> \
    --num-gpus N \
    --eval-only \
    train.output_dir=<output_dir> \
    train.init_checkpoint=<output_dir>/meanmodel.pth \
```

- For Swin-L based models, set the weight decay as 0.05:
```
python projects/mr_detr_align/train_net_swin.py \
    --config-file <config-file> \
    --num-gpus N \
    dataloader.train.total_batch_size=16 \
    train.output_dir=<output_dir> \
    train.amp.enabled=True \ # mixed precision training
    model.transformer.encoder.use_checkpoint=True \ # gradient checkpointing, save gpu memory but lower speed

# to get mean model, which is more stable than ema, and improves about 0.1~0.2%.
python projects/modelmean_12ep.py --folder <output_dir>
python projects/modelmean_24ep.py --folder <output_dir>

python projects/train_net.py \
    --config-file <config-file> \
    --num-gpus N \
    --eval-only \
    train.output_dir=<output_dir> \
    train.init_checkpoint=<output_dir>/meanmodel.pth \
```


## Evaluate
```
python projects/train_net.py \
    --config-file <config_file> \
    --eval-only \
    --num-gpus=4 \
    train.init_checkpoint=<checkpoint_path> \
```


## Citation
```
@inproceedings{zhang2024mr,
  title={Mr. DETR: Instructive Multi-Route Training for Detection Transformers},
  author={Zhang, Chang-Bin and Zhong, Yujie and Han, Kai},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```



