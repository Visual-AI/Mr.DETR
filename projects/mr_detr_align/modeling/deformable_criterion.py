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
        self.rank_weight = None
    
    
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
                indices = self.topk_matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes * 3, userank=True, **kwargs)
                    l_dict = {k + f"_group_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        if "sep" in outputs:
            for i, aux_outputs in enumerate(outputs["sep"]):
                indices = self.topk_matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes * 3, userank=True, **kwargs)
                    l_dict = {k + f"_sep_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        

        # Compute losses for two-stage deformable-detr
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])
                
            # NOTE refer to the implementation of MS-DETR
            # enc_outputs['anchors'], enc_outputs['pred_boxes'] = enc_outputs['pred_boxes'], enc_outputs['anchors']
            indices = self.matcher(enc_outputs, bin_targets)
            # enc_outputs['anchors'], enc_outputs['pred_boxes'] = enc_outputs['pred_boxes'], enc_outputs['anchors']
            
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
            # enc_outputs['anchors'], enc_outputs['pred_boxes'] = enc_outputs['pred_boxes'], enc_outputs['anchors']
            indices = self.matcher(enc_outputs, bin_targets)
            # enc_outputs['anchors'], enc_outputs['pred_boxes'] = enc_outputs['pred_boxes'], enc_outputs['anchors']
            
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + "_enc_o2m": v for k, v in l_dict.items()}
                losses.update(l_dict)
            return losses
        
    def loss_labels(self, outputs, targets, indices, num_boxes, **kwargs):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        src_boxes = outputs['pred_boxes'][idx].float()
        target_boxes = torch.cat([t['boxes'][i].float() for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()

        alpha, gamma = 0.25, 2
        
        # Ensure pred_score is in FP32 for operations that require higher precision
        pred_score = torch.sigmoid(src_logits)
        
        pos_weights = torch.zeros_like(src_logits)
        neg_weights = pred_score ** gamma
        pos_ind = [id for id in idx]
        pos_ind.append(target_classes_o)
        
        # Ensure t is in FP32 for operations that require higher precision
        t = pred_score[pos_ind].pow(alpha) * ious.pow(1 - alpha)
        t = torch.clamp(t, 0.01).detach()
        tau = 1.5
        if 'userank' in kwargs and kwargs['userank']:
            rank = self.get_local_rank_o2m(t, indices)
            rank_weight = torch.exp(-rank/tau).cuda()
        else:
            rank_weight = 1
        
        t = t * rank_weight
        pos_weights[pos_ind] = t
        neg_weights[pos_ind] = 1 - t 
        
        # Ensure loss_ce is calculated in FP32
        loss_ce = - pos_weights * pred_score.log() - neg_weights * (1 - pred_score).log()
        loss_ce = loss_ce.sum().float() / num_boxes
        losses = {"loss_class": loss_ce}
        self.rank_weight = rank_weight
    
        return losses

    
    def loss_boxes(self, outputs, targets, indices, num_boxes, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx].float()
        target_boxes = torch.cat([t["boxes"][i].float() for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = (loss_giou * self.rank_weight).sum() / num_boxes

        return losses
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "class": self.loss_labels,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    
    def get_local_rank_o2m(self, quality, indices):
        #quality: one-dimension tensor 
        #indices: matching result
        bs = len(indices)
        device = quality.device
        tgt_size = [len(tgt_ind) for _,tgt_ind in indices]
        ind_start = 0
        rank_list = []
        for i in range(bs):
            if  tgt_size[i] == 0:
                rank_list.append(torch.zeros(0,dtype=torch.long,device=device))
                continue     
            num_tgt = max(indices[i][1]) + 1
            # split quality of one item
            quality_per_img = quality[ind_start:ind_start+tgt_size[i]]
            ind_start += tgt_size[i]

            rank_per_img = torch.zeros(len(indices[i][1]),dtype=torch.long,).cuda()

            int_unique, int_unique_count = indices[i][1].unique(return_counts=True)
            for j, id in enumerate(int_unique):
                if int_unique_count[j] == 1:
                    k_indices = torch.where(indices[i][1] == id)[0]
                    rank_per_img[k_indices] = 0
                if int_unique_count[j] > 1:
                    # ForkedPdb().set_trace()
                    k_indices = torch.where(indices[i][1] == id)[0]
                    k_indices = k_indices[torch.sort(quality_per_img[k_indices], descending=True)[1]]
                    keep_k = torch.arange(int_unique_count[j])
                    rank_per_img[k_indices] = keep_k.cuda()
            rank_list.append(rank_per_img.flatten())
        return torch.cat(rank_list, 0)