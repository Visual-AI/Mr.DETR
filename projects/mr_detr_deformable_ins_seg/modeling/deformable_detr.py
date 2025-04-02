# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import inverse_sigmoid

from detectron2.structures import Boxes, ImageList, Instances
from pycocotools import mask as coco_mask

from detectron2.structures import Instances, ROIMasks
from fairscale.nn.checkpoint import checkpoint_wrapper
from detectron2.layers.nms import batched_nms


# perhaps should rename to "resize_instance"
def detector_postprocess(
    results: Instances, output_height: int, output_width: int, mask_threshold: float = 0.5
):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.
    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    if isinstance(output_width, torch.Tensor):
        # This shape might (but not necessarily) be tensors during tracing.
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances(new_size, **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]
    # if results.has("pred_masks"):
    #     if isinstance(results.pred_masks, ROIMasks):
    #         roi_masks = results.pred_masks
    #     else:
    #         # pred_masks is a tensor of shape (N, 1, M, M)
    #         roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
    #     results.pred_masks = roi_masks.to_bitmasks(
    #         results.pred_boxes, output_height, output_width, mask_threshold
    #     ).tensor  # TODO return ROIMasks/BitMask object in the future

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results

class DeformableDETR(nn.Module):
    """Implements the Deformable DETR model.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2010.04159>`_ .

    Args:
        backbone (nn.Module): the backbone module.
        position_embedding (nn.Module): the position embedding module.
        neck (nn.Module): the neck module.
        transformer (nn.Module): the transformer module.
        embed_dim (int): the dimension of the embedding.
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): whether to use auxiliary loss. Default: True.
        with_box_refine (bool): whether to use box refinement. Default: False.
        as_two_stage (bool): whether to use two-stage. Default: False.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 100.

    """

    def __init__(
        self,
        backbone,
        position_embedding,
        neck,
        transformer,
        embed_dim,
        num_classes,
        num_queries,
        criterion,
        pixel_mean,
        pixel_std,
        aux_loss=True,
        with_box_refine=False,
        as_two_stage=False,
        select_box_nums_for_evaluation=100,
        device="cuda",
        mixed_selection=False, # tricks
        use_checkpoint=False,
    ):
        super().__init__()
        # define backbone and position embedding module
        self.backbone = backbone
        self.position_embedding = position_embedding

        # define neck module
        self.neck = neck

        # define learnable query embedding
        self.num_queries = num_queries
        if not as_two_stage:
            self.query_embedding = nn.Embedding(num_queries, embed_dim * 2)
        elif mixed_selection:
            self.query_embedding = nn.Embedding(num_queries, embed_dim)

        # define transformer module
        self.transformer = transformer

        # define classification head and box head
        self.num_classes = num_classes
        self.class_embed = nn.Linear(embed_dim, num_classes)
                                
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)

        # where to calculate auxiliary loss in criterion
        self.aux_loss = aux_loss
        self.criterion = criterion

        # define contoller for box refinement and two-stage variants
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage

        # init parameters for heads
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value 
        
        
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)

        # If two-stage, the last class_embed and bbox_embed is for region proposal generation
        # Decoder layers share the same heads without box refinement, while use the different
        # heads when box refinement is used.
        num_pred = (
            (transformer.decoder.num_layers + 1) if as_two_stage else transformer.decoder.num_layers
        )
        if with_box_refine:
            self.class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed) for i in range(num_pred)]
            )
            self.bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for i in range(num_pred)]
            )

            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        # hack implementation for two-stage. The last class_embed and bbox_embed is for region proposal generation
        if as_two_stage:
            self.transformer.decoder.class_embed = self.class_embed
            self.transformer.decoder.bbox_embed_enc_o2m = copy.deepcopy(self.bbox_embed[0])
            # self.transformer.decoder.bbox_embed_enc_sep = copy.deepcopy(self.bbox_embed[0])
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

            self.transformer.decoder.class_embed_o2m_encoder = copy.deepcopy(self.transformer.decoder.class_embed[0])
            # self.transformer.decoder.class_embed_sep_encoder = copy.deepcopy(self.transformer.decoder.class_embed[0])

        # set topk boxes selected for inference
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation

        # normalizer for input raw images
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        
        self.mixed_selection = mixed_selection
        
        num_heads, hidden_dim = 8, 256
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, num_heads, dropout=0)
        self.mask_head = MaskHeadSmallConv(hidden_dim + num_heads, [1024, 512, 256], hidden_dim)
        input_proj_list = []
        for in_channels in [1024, 512, 256]:
            input_proj_list.append(nn.Conv2d(2*in_channels, in_channels, kernel_size=3, stride=2, padding=1))
        self.seg_input_proj = nn.ModuleList(input_proj_list)
        
        
        # for layer in self.backbone.stages:
        #     layer = checkpoint_wrapper(layer)
        # for layer in self.seg_input_proj:
        #     layer = checkpoint_wrapper(layer)
        

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)

        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                # mask padding regions in batched images
                img_h, img_w = batched_inputs[img_id]["instances"].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)

        # original features
        features = self.backbone(images.tensor)  # output feature dict

        # project backbone features to the reuired dimension of transformer
        # we use multi-scale features in deformable DETR
        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        # initialize object query embeddings
        query_embeds = None
        if not self.as_two_stage or self.mixed_selection:
            query_embeds = self.query_embedding.weight
        

        
        
        (
            inter_states,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
            outputs_o2m, # NOTE
            anchors,
            outputs_sep, # NOTE
            enc_memory
        ) = self.transformer(
            multi_level_feats, multi_level_masks, multi_level_position_embeddings, query_embeds
        )
        

        # Calculate output coordinates and classes.
        outputs_classes = []
        outputs_coords = []
        
        outputs_classes_o2m = []
        outputs_coords_o2m = []
            
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid() 
            
            if self.training == False:
                outputs_class = outputs_class.sigmoid()
                if len(outputs_classes) > 0:
                    outputs_class = (outputs_class ** 2.001 + outputs_classes[-1].sigmoid() ** 1.853) / 2.
                    outputs_class = torch.pow(outputs_class, 1./2)
                outputs_class = inverse_sigmoid(outputs_class)       
                
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        
        
        outputs_coords_o2m, outputs_classes_o2m = [], []
        inter_states_o2m, inter_references_o2m = outputs_o2m['inter_states_o2m'], outputs_o2m['inter_references_o2m']
        for lvl in range(inter_states_o2m.shape[0]):
            if lvl == 0:
                reference = outputs_o2m["init_reference_points"]
            else:
                reference = inter_references_o2m[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states_o2m[lvl])
            tmp = self.bbox_embed[lvl](inter_states_o2m[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()      
                
            outputs_classes_o2m.append(outputs_class)
            outputs_coords_o2m.append(outputs_coord)
        
        
        outputs_coords_sep, outputs_classes_sep = [], []
        inter_states_sep, inter_references_sep = outputs_sep['inter_states_o2m'], outputs_sep['inter_references_o2m']
        for lvl in range(inter_states_sep.shape[0]):
            if lvl == 0:
                reference = outputs_sep["init_reference_points"]
            else:
                reference = inter_references_sep[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states_sep[lvl])
            tmp = self.bbox_embed[lvl](inter_states_sep[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()      
                
            outputs_classes_sep.append(outputs_class)
            outputs_coords_sep.append(outputs_coord)
        
        
        outputs_class = torch.stack(outputs_classes)
        # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
        outputs_coord = torch.stack(outputs_coords)
        # tensor shape: [num_decoder_layers, bs, num_query, 4]
        outputs_class_o2m = torch.stack(outputs_classes_o2m)
        outputs_coord_o2m = torch.stack(outputs_coords_o2m)
        
        outputs_class_sep = torch.stack(outputs_classes_sep)
        outputs_coord_sep = torch.stack(outputs_coords_sep)

        # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
        output["group"] = [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class_o2m, outputs_coord_o2m)
            # for a, b in zip(outputs_class[:, :, 150:], outputs_coord[:, :, 150:])
        ]
        output["sep"] = [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class_sep, outputs_coord_sep)
            # for a, b in zip(outputs_class[:, :, 150:], outputs_coord[:, :, 150:])
        ]
        
        # get instance mask prediction
        seg_mask = multi_level_masks[-1]
        last_level_memory = enc_memory[:, -seg_mask.shape[1] * seg_mask.shape[2]:, :]
        last_level_memory = last_level_memory.reshape(-1, seg_mask.shape[1], seg_mask.shape[2], last_level_memory.shape[2])
        last_level_memory = last_level_memory.permute(0, 3, 1, 2)
        
        bbox_mask = self.bbox_attention(inter_states[-1], last_level_memory, mask=seg_mask)
        fpns_features = [self.seg_input_proj[0](features["res5"]), self.seg_input_proj[1](features["res4"]), self.seg_input_proj[2](features["res3"])]
        seg_masks = self.mask_head(last_level_memory, bbox_mask, fpns_features)
        outputs_seg_masks = seg_masks.view(last_level_memory.shape[0], 300, seg_masks.shape[-2], seg_masks.shape[-1])
        output["pred_masks"] = outputs_seg_masks
        
        bbox_mask = self.bbox_attention(inter_states_o2m[-1], last_level_memory, mask=seg_mask)
        seg_masks = self.mask_head(last_level_memory, bbox_mask, fpns_features)
        outputs_seg_masks = seg_masks.view(last_level_memory.shape[0], 300, seg_masks.shape[-2], seg_masks.shape[-1])
        output["group"][-1]["pred_masks"] = outputs_seg_masks
        
        bbox_mask = self.bbox_attention(inter_states_sep[-1], last_level_memory, mask=seg_mask)
        seg_masks = self.mask_head(last_level_memory, bbox_mask, fpns_features)
        outputs_seg_masks = seg_masks.view(last_level_memory.shape[0], 300, seg_masks.shape[-2], seg_masks.shape[-1])
        output["sep"][-1]["pred_masks"] = outputs_seg_masks
        
        

        if self.as_two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            output["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
                "anchors": anchors.sigmoid()
            }
            output["enc_outputs_o2m"] = {
                "pred_logits": outputs_o2m["enc_outputs_class_o2m"],
                "pred_boxes": outputs_o2m["enc_outputs_box_o2m"].sigmoid(),
                "anchors": outputs_o2m["anchors"].sigmoid()
            }
            # output["enc_outputs_sep"] = {
            #     "pred_logits": outputs_sep["enc_outputs_class_o2m"],
            #     "pred_boxes": outputs_sep["enc_outputs_box_o2m"].sigmoid(),
            #     "anchors": outputs_sep["anchors"].sigmoid()
            # }

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, images.tensor.shape[-2], images.tensor.shape[-1])
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            mask_pred = output["pred_masks"]            
            results = self.inference(box_cls, box_pred, mask_pred, images.image_sizes)
            
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                pred_mask = torch.nn.functional.interpolate(
                    results_per_image.pred_masks.unsqueeze(1),
                    (height, width),
                    mode='bilinear', align_corners=False)
                pred_mask = (pred_mask[:, 0] > 0).float()
                r = detector_postprocess(results_per_image, height, width)
                r.pred_masks = pred_mask
                processed_results.append({"instances": r})
            return processed_results

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def nms_inference_wo_mask(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        bs, n_queries, n_cls = box_cls.shape

        # Select top-k confidence boxes for inference
        prob = box_cls.sigmoid()

        all_scores = prob.view(bs, n_queries * n_cls).to(box_cls.device)
        all_indexes = torch.arange(n_queries * n_cls)[None].repeat(bs, 1).to(box_cls.device)
        all_boxes = torch.div(all_indexes, box_cls.shape[2], rounding_mode="floor")
        all_labels = all_indexes % box_cls.shape[2]

        # convert to xyxy for nms post-process
        boxes = box_cxcywh_to_xyxy(box_pred)
        boxes = torch.gather(boxes, 1, all_boxes.unsqueeze(-1).repeat(1, 1, 4))

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
            zip(all_scores, all_labels, boxes, image_sizes)
        ):

            pre_topk = scores_per_image.topk(10000).indices
            box = box_pred_per_image[pre_topk]
            score = scores_per_image[pre_topk]
            label = labels_per_image[pre_topk]

            # nms post-process
            keep_index = batched_nms(box, score, label, 0.7)[:self.select_box_nums_for_evaluation]
            keep_index = keep_index[:self.select_box_nums_for_evaluation]
            result = Instances(image_size)
            result.pred_boxes = Boxes(box[keep_index])
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = score[keep_index]
            result.pred_classes = label[keep_index]
            results.append(result)
        return results
    def inference_wo_mask(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # Select top-k confidence boxes for inference
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
            zip(scores, labels, boxes, image_sizes)
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results
    def nms_inference(self, box_cls, box_pred, mask_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        bs, n_queries, n_cls = box_cls.shape

        # Select top-k confidence boxes for inference
        prob = box_cls

        all_scores = prob.view(bs, n_queries * n_cls).to(box_cls.device)
        all_indexes = torch.arange(n_queries * n_cls)[None].repeat(bs, 1).to(box_cls.device)
        all_boxes = torch.div(all_indexes, box_cls.shape[2], rounding_mode="floor")
        all_labels = all_indexes % box_cls.shape[2]

        # convert to xyxy for nms post-process
        boxes = box_cxcywh_to_xyxy(box_pred)
        boxes = torch.gather(boxes, 1, all_boxes.unsqueeze(-1).repeat(1, 1, 4))
        masks = torch.gather(mask_pred, 1, all_boxes.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, mask_pred.shape[-2], mask_pred.shape[-1]))
        for i, (scores_per_image, labels_per_image, box_pred_per_image, mask_pred_per_image, image_size) in enumerate(
            zip(all_scores, all_labels, boxes, masks, image_sizes)
        ):

            pre_topk = scores_per_image.topk(300*2).indices
            box = box_pred_per_image[pre_topk]
            score = scores_per_image[pre_topk]
            label = labels_per_image[pre_topk]
            mask = mask_pred_per_image[pre_topk]

            # nms post-process
            keep_index = batched_nms(box, score, label, 0.7)
            keep_index = keep_index[:self.select_box_nums_for_evaluation]

            result = Instances(image_size)
            result.pred_boxes = Boxes(box[keep_index])
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = score[keep_index]
            result.pred_classes = label[keep_index]
            result.pred_masks = mask[keep_index]
            results.append(result)
        return results
    
    def inference(self, box_cls, box_pred, mask_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # Select top-k confidence boxes for inference
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        masks = torch.gather(mask_pred, 1, topk_boxes.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, mask_pred.shape[-2], mask_pred.shape[-1]))
        for i, (scores_per_image, labels_per_image, box_pred_per_image, mask_pred_per_image, image_size) in enumerate(
            zip(scores, labels, boxes, masks, image_sizes)
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            result.pred_masks = mask_pred_per_image
            results.append(result)
        return results

    def prepare_targets(self, targets, h_pad, w_pad):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            # prepare masks
            gt_masks = targets_per_image.gt_masks
            gt_masks = self.convert_coco_poly_to_mask(gt_masks, h, w)
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes, "masks":padded_masks })
        return new_targets
    
    from pycocotools import mask as coco_mask


    def convert_coco_poly_to_mask(self, segmentations, height, width):
        masks = []
        for polygons in segmentations:
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            mask = mask.any(dim=2)
            masks.append(mask)
        if masks:
            masks = torch.stack(masks, dim=0)
        else:
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
        return masks

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images



class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, bbox_mask, fpns):
        def expand(tensor, length):
            return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)

        x = torch.cat([expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        x = self.out_lay(x)
        return x


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask=None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
        weights = self.dropout(weights)
        return weights