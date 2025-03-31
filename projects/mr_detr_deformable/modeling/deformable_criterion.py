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
from typing import List
import torch
import torch.nn.functional as F

from detrex.modeling import SetCriterion
from detrex.utils import get_world_size, is_dist_avail_and_initialized
from detrex.layers import box_cxcywh_to_xyxy, box_iou, generalized_box_iou


class DeformableCriterion(SetCriterion):
    """This class computes the loss for Deformable-DETR
    and two-stage Deformable-DETR
    """

    def __init__(
        self,
        num_classes,
        matcher,
        enc_matcher,
        weight_dict,
        losses: List[str] = ["class", "boxes"],
        eos_coef: float = 0.1,
        loss_class_type: str = "focal_loss",
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super(DeformableCriterion, self).__init__(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            eos_coef=eos_coef,
            loss_class_type=loss_class_type,
            alpha=alpha,
            gamma=gamma,
        )
        from .one2manyMatcherDAC import Stage2Assigner
        self.topk_matcher = Stage2Assigner(num_queries=900, max_k=6)
        self.enc_matcher = enc_matcher
    
    
    @staticmethod
    def indices_merge(num_queries, o2o_indices, o2m_indices):
        bs = len(o2o_indices)
        temp_indices = torch.zeros(bs, num_queries, dtype=torch.int64).cuda() - 1
        new_one2many_indices = []

        for i in range(bs):
            one2many_fg_inds = o2m_indices[i][0].cuda()
            one2many_gt_inds = o2m_indices[i][1].cuda()
            one2one_fg_inds = o2o_indices[i][0].cuda()
            one2one_gt_inds = o2o_indices[i][1].cuda()

            combined = torch.cat((torch.stack((one2many_fg_inds, one2many_gt_inds), dim=1), torch.stack((one2one_fg_inds, one2one_gt_inds), dim=1)))
            unique_pairs = torch.unique(combined, dim=0)
            fg_inds, gt_inds = unique_pairs[:, 0], unique_pairs[:, 1]
            
            # temp_indices[i][one2one_fg_inds] = one2one_gt_inds
            # temp_indices[i][one2many_fg_inds] = one2many_gt_inds
            # fg_inds = torch.nonzero(temp_indices[i] >= 0).squeeze(1)
            # gt_inds = temp_indices[i][fg_inds]
            new_one2many_indices.append((fg_inds, gt_inds))

        return new_one2many_indices
   
    def forward(self, outputs, targets):
        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != "aux_outputs" and k != "enc_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        if "group" in outputs:
            for i, aux_outputs in enumerate(outputs["group"]):
                # o2o_indices = self.matcher(aux_outputs, targets)
                # o2m_indices = self.topk_matcher(aux_outputs, targets)
                # indices = self.indices_merge(900, o2o_indices, o2m_indices)
                indices = self.topk_matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_group_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        if "sep" in outputs:
            for i, aux_outputs in enumerate(outputs["sep"]):
                indices = self.topk_matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_sep_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        

        # Compute losses for two-stage deformable-detr
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])
                
            # NOTE refer to the implementation of MS-DETR
            # NOTE I don't know if it can improve the performance in Mr.DETR but use it
            enc_outputs['anchors'], enc_outputs['pred_boxes'] = enc_outputs['pred_boxes'], enc_outputs['anchors']
            indices = self.enc_matcher(enc_outputs, bin_targets)
            enc_outputs['anchors'], enc_outputs['pred_boxes'] = enc_outputs['pred_boxes'], enc_outputs['anchors']
            
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)
        
        
            # enc o2m outputs loss
            enc_outputs = outputs["enc_outputs_o2m"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])
                
            # NOTE refer to the implementation of MS-DETR
            # NOTE I don't know if it can improve the performance in Mr.DETR but use it
            enc_outputs['anchors'], enc_outputs['pred_boxes'] = enc_outputs['pred_boxes'], enc_outputs['anchors']
            indices = self.enc_matcher(enc_outputs, bin_targets)
            enc_outputs['anchors'], enc_outputs['pred_boxes'] = enc_outputs['pred_boxes'], enc_outputs['anchors']
            
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + "_enc_o2m": v for k, v in l_dict.items()}
                losses.update(l_dict)
        
        
            # enc o2m outputs loss
            # enc_outputs = outputs["enc_outputs_sep"]
            # bin_targets = copy.deepcopy(targets)
            # for bt in bin_targets:
            #     bt["labels"] = torch.zeros_like(bt["labels"])
                
            # # NOTE refer to the implementation of MS-DETR
            # enc_outputs['anchors'], enc_outputs['pred_boxes'] = enc_outputs['pred_boxes'], enc_outputs['anchors']
            # indices = self.enc_matcher(enc_outputs, bin_targets)
            # enc_outputs['anchors'], enc_outputs['pred_boxes'] = enc_outputs['pred_boxes'], enc_outputs['anchors']
            
            # for loss in self.losses:
            #     l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
            #     l_dict = {k + "_enc_sep": v for k, v in l_dict.items()}
            #     losses.update(l_dict)
       
        return losses