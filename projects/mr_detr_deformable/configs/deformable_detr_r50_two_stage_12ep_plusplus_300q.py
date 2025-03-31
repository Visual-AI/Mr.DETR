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

model.select_box_nums_for_evaluation=300
train.checkpointer.period = 10000

dataloader.train.num_workers=16
model.num_queries=300
# model.transformer.encoder.use_checkpoint=True