from .deformable_detr_r50_12ep import train, dataloader, optimizer, lr_multiplier, model
from detectron2.modeling.backbone import SwinTransformer
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec


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
train.checkpointer.period = 10000
# model.transformer.encoder.use_checkpoint=True

dataloader.train.num_workers=8

model.backbone = L(SwinTransformer)(
    pretrain_img_size=384,
    embed_dim=192,
    depths=(2, 2, 18, 2),
    num_heads=(6, 12, 24, 48),
    window_size=12,
    out_indices=(1, 2, 3),
)

# modify neck config
model.neck.input_shapes = {
    "p1": ShapeSpec(channels=384),
    "p2": ShapeSpec(channels=768),
    "p3": ShapeSpec(channels=1536),
}
model.neck.in_features = ["p1", "p2", "p3"]
train.init_checkpoint = "output/swin_large_patch4_window12_384_22k.pth"