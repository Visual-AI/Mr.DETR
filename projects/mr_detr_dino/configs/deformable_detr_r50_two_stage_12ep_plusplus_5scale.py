from .deformable_detr_r50_12ep import train, dataloader, optimizer, lr_multiplier, model

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


model.select_box_nums_for_evaluation = 100

# model.transformer.encoder.use_checkpoint=True

dataloader.train.num_workers=8


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