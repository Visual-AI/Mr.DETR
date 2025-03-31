from .deformable_detr_r50_12ep import train, dataloader, optimizer, model

# modify model config
model.with_box_refine = True
model.as_two_stage = True

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
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

model.transformer.encoder.use_checkpoint=True
dataloader.train.num_workers=8
train.max_iter = 90000 * 2
train.eval_period = 7500 * 2
train.log_period = 200
train.checkpointer.period = 10000


from detectron2.layers import ShapeSpec
# modify model config to generate 4 scale backbone features 
# and 5 scale input features
model.backbone.out_features = ["res2", "res3", "res4", "res5"]

model.neck.input_shapes = {
    "res2": ShapeSpec(channels=256),
    "res3": ShapeSpec(channels=512),
    "res4": ShapeSpec(channels=1024),
    "res5": ShapeSpec(channels=2048),
}
model.neck.in_features = ["res2", "res3", "res4", "res5"]
model.neck.num_outs = 5
model.transformer.num_feature_levels = 5


from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler
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

lr_multiplier = custom_coco_pretrain_scheduler(epochs=12, decay_epochs=10, warmup_epochs=0, batch_size=8)
