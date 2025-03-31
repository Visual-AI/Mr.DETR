from .deformable_detr_r50_12ep import train, dataloader, optimizer, model
from detectron2.modeling.backbone import SwinTransformer
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
def custom_coco_pretrain_scheduler(epochs=12, decay_epochs=10, warmup_epochs=0, batch_size=16):
    """
    Returns the config for a default multi-step LR scheduler such as "50epochs",
    commonly referred to in papers, where every 1x has the total length of 1440k
    training images (~12 COCO epochs). LR is decayed once at the end of training.

    Args:
        epochs (int): total training epochs.
        decay_epochs (int): lr decay steps.
        warmup_epochs (int): warmup epochs.

    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    # 1803392
    total_steps = epochs * 7500 * 16 // batch_size    
    decay_steps = decay_epochs * 7500 * 16 // batch_size
    warmup_steps = warmup_epochs * 7500 * 16 // batch_size
    scheduler = L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[decay_steps, total_steps],
    )
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / total_steps,
        warmup_method="linear",
        warmup_factor=0.001,
    )

# modify model config
model.with_box_refine = True
model.as_two_stage = True

# modify training config
train.output_dir = "./output/deformable_detr_r50_two_stage_12ep"

model.transformer.encoder.feedforward_dim=2048
model.transformer.decoder.feedforward_dim=2048

model.transformer.encoder.attn_dropout=0.0
model.transformer.encoder.ffn_dropout=0.0
model.transformer.decoder.attn_dropout=0.0
model.transformer.decoder.ffn_dropout=0.0

model.mixed_selection = True
model.transformer.mixed_selection = True
model.transformer.decoder.look_forward_twice = True


model.select_box_nums_for_evaluation = 300
model.num_queries = 900
model.transformer.encoder.use_checkpoint=True

dataloader.train.num_workers=8

model.backbone = L(SwinTransformer)(
    pretrain_img_size=384,
    embed_dim=192,
    depths=(2, 2, 18, 2),
    num_heads=(6, 12, 24, 48),
    window_size=12,
    out_indices=(0, 1, 2, 3),
    drop_path_rate=0.2, # default 0.2
)

# modify neck config
model.neck.input_shapes = {
    "p0": ShapeSpec(channels=192),
    "p1": ShapeSpec(channels=384),
    "p2": ShapeSpec(channels=768),
    "p3": ShapeSpec(channels=1536),
}
model.backbone.use_checkpoint = True
model.neck.in_features = ["p0", "p1", "p2", "p3"]
model.neck.num_outs = 5
model.transformer.num_feature_levels = 5

train.init_checkpoint="work_dirs/Objects365_swinl_5scale_12ep/model_2194999.pth"

import warnings
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly.")




import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detrex.data import DetrDatasetMapper

dataloader.train.mapper=L(DetrDatasetMapper)(
    augmentation=[
        L(T.RandomFlip)(),
        L(T.ResizeShortestEdge)(
            short_edge_length=(
                480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800,
                832, 864, 896, 928, 960, 992, 1024, 1056, 1088, 1120,
                1152, 1184, 1216, 1248, 1280, 1312, 1344, 1376, 1408, 1440,
                1472, 1504, 1536
            ),
            max_size=2400,
            sample_style="choice",
        ),
    ],
    augmentation_with_crop=[
        L(T.RandomFlip)(),
        L(T.ResizeShortestEdge)(
            short_edge_length=(400, 500, 600),
            sample_style="choice",
        ),
        L(T.RandomCrop)(
            crop_type="absolute_range",
            crop_size=(384, 600),
        ),
        L(T.ResizeShortestEdge)(
            short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, \
                               832, 864, 896, 928, 960, 992, 1024, 1056, 1088, 1120, \
                               1152, 1184, 1216, 1248, 1280, 1312, 1344, 1376, 1408, 1440,\
                                1472, 1504, 1536),
            max_size=2400,
            sample_style="choice",
        ),
    ],
    is_train=True,
    mask_on=False,
    img_format="RGB",
)

dataloader.test.mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=1280,
                max_size=2048,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    )

train.amp.enabled=True
train.max_iter = 90000 * 4
train.eval_period = 7500 * 4
train.log_period = 400
train.checkpointer.period = 7500

lr_multiplier = custom_coco_pretrain_scheduler(epochs=12, decay_epochs=10, warmup_epochs=0, batch_size=4)


