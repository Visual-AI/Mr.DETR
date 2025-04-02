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
# ------------------------------------------------------------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/bricks/transformer.py
# ------------------------------------------------------------------------------------------------

import copy
import warnings
from typing import List
import torch
import torch.nn as nn
from typing import Optional



class CustomMultiheadAttention(nn.Module):
    """A wrapper for ``torch.nn.MultiheadAttention``

    Implemente MultiheadAttention with identity connection,
    and position embedding is also passed as input.

    Args:
        embed_dim (int): The embedding dimension for attention.
        num_heads (int): The number of attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `MultiheadAttention`.
            Default: 0.0.
        batch_first (bool): if `True`, then the input and output tensor will be
            provided as `(bs, n, embed_dim)`. Default: False. `(n, bs, embed_dim)`
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(CustomMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=batch_first,
            **kwargs,
        )

        self.proj_drop = nn.Dropout(proj_drop)
        
        # self.instruction_o2o = nn.Embedding(10, embed_dim)
        # self.instruction_o2o_pos = nn.Embedding(10, embed_dim)
        self.instruction_o2m = nn.Embedding(10, embed_dim)
        self.instruction_o2m_pos = nn.Embedding(10, embed_dim)
        
        self.norm_o2o = nn.LayerNorm(embed_dim)
        self.norm_o2m = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward function for `MultiheadAttention`

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_query, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_key, embed_dim)`
            value (torch.Tensor): Value embeddings with the same shape as `key`.
                Same in `torch.nn.MultiheadAttention.forward`. Default: None.
                If None, the `key` will be used.
            identity (torch.Tensor): The tensor, with the same shape as x, will
                be used for identity addition. Default: None.
                If None, `query` will be used.
            query_pos (torch.Tensor): The position embedding for query, with the
                same shape as `query`. Default: None.
            key_pos (torch.Tensor): The position embedding for key. Default: None.
                If None, and `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`.
            attn_mask (torch.Tensor): ByteTensor mask with shape `(num_query, num_key)`.
                Same as `torch.nn.MultiheadAttention.forward`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor with shape `(bs, num_key)` which
                indicates which elements within `key` to be ignored in attention.
                Default: None.
        """
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(
                        f"position encoding of key is" f"missing in {self.__class__.__name__}."
                    )
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        if "o2m" in list(kwargs.keys()) and kwargs["o2m"]:
            instruction = self.instruction_o2m.weight + self.instruction_o2m_pos.weight
            instruction = instruction[None].repeat(query.shape[0], 1, 1)
            query = torch.cat([instruction, query], dim=1)
            key = torch.cat([instruction, key], dim=1)
            value = torch.cat([
                self.instruction_o2m.weight[None].repeat(query.shape[0], 1, 1),
                value,
            ], dim=1)
            out = self.attn(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )[0]
            out = out[:, 10:]
            out = self.proj_drop(out) + identity
            out = self.norm_o2m(out)
        else:
            # instruction = self.instruction_o2o.weight + self.instruction_o2o_pos.weight
            # instruction = instruction[None].repeat(query.shape[0], 1, 1)
            # query = torch.cat([instruction, query], dim=1)
            # key = torch.cat([instruction, key], dim=1)
            # value = torch.cat([
            #     self.instruction_o2o.weight[None].repeat(query.shape[0], 1, 1),
            #     value,
            # ], dim=1)
            out = self.attn(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )[0]
            # out = out[:, 10:]
            out = identity + self.proj_drop(out)
            out = self.norm_o2o(out)
        return out



class BaseTransformerLayer(nn.Module):
    def __init__(
        self,
        attn: List[nn.Module],
        ffn: nn.Module,
        norm: nn.Module,
        operation_order: tuple = None,
    ):
        super(BaseTransformerLayer, self).__init__()
        assert set(operation_order).issubset({"self_attn", "norm", "cross_attn", "ffn"})

        # count attention nums
        num_attn = operation_order.count("self_attn") + operation_order.count("cross_attn")
        if isinstance(attn, nn.Module):
            attn = [copy.deepcopy(attn) for _ in range(num_attn)]
        else:
            assert len(attn) == num_attn, (
                f"The length of attn (nn.Module or List[nn.Module]) {num_attn}"
                f"is not consistent with the number of attention in "
                f"operation_order {operation_order}"
            )

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.pre_norm = operation_order[0] == "norm"
        self.attentions = nn.ModuleList()
        index = 0
        for operation_name in operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                self.attentions.append(attn[index])
                index += 1

        self.embed_dim = self.attentions[0].embed_dim

        # count ffn nums
        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count("ffn")
        for _ in range(num_ffns):
            self.ffns.append(copy.deepcopy(ffn))

        # count norm nums
        self.norms = nn.ModuleList()
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(copy.deepcopy(norm))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        query_pos: torch.Tensor = None,
        key_pos: torch.Tensor = None,
        attn_masks: List[torch.Tensor] = None,
        query_key_padding_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        **kwargs,
    ):
        """Forward function for `BaseTransformerLayer`.

        **kwargs contains the specific arguments of attentions.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` or `(bs, num_query, embed_dim)`
                which should be specified follows the attention module used in
                `BaseTransformerLayer`.
            key (torch.Tensor): Key embeddings used in `Attention`.
            value (torch.Tensor): Value embeddings with the same shape as `key`.
            query_pos (torch.Tensor): The position embedding for `query`.
                Default: None.
            key_pos (torch.Tensor): The position embedding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): A list of 2D ByteTensor used
                in calculation the corresponding attention. The length of
                `attn_masks` should be equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape `(bs, num_query)`. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (torch.Tensor): ByteTensor for `key`, with
                shape `(bs, num_key)`. Default: None.
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f"Use same attn_mask in all attentions in " f"{self.__class__.__name__} ")
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        # for layer_idx, layer in enumerate(self.operation_order):
        layer_idx = 0
        while layer_idx < len(self.operation_order):
            layer = self.operation_order[layer_idx]
            layer_idx += 1
            if layer == "self_attn":
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                
                attn_index += 1
                identity = query
                    

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                if 'sep' in kwargs and kwargs['sep']:
                    query = self.ffns[ffn_index](query, identity if self.pre_norm else None, **kwargs)
                else:
                    query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1
        
            

        return query
    




import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class CustomLinear(nn.Module):
    def __init__(self, embed_dim, output_dim, bias, k_groups):
        super().__init__()
        self.linear = nn.Linear(embed_dim, output_dim, bias=bias)
        self.linear_sep = nn.Linear(embed_dim, output_dim, bias=bias)
        
                
    def forward(self, x, **kwargs):
        if 'sep' in kwargs and kwargs['sep']:
            return self.linear_sep(x)
        else:
            return self.linear(x)
        
        
        

class CustomBlock(nn.Module):
    def __init__(self, embed_dim, output_dim, bias, ffn_drop, activation, k_groups):
        super().__init__()
        self.linear = CustomLinear(embed_dim, output_dim, bias=bias, k_groups=k_groups)
        self.activation = activation
        self.dropout = nn.Dropout(ffn_drop)
        
        
    def forward(self, x, **kwargs):
        x = self.linear(x, **kwargs)
        x = self.activation(x)
        x = self.dropout(x)
        return x



class DynamicFFN(nn.Module):
    """The implementation of feed-forward networks (FFNs)
    with identity connection.

    Args:
        embed_dim (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_dim (int): The hidden dimension of FFNs.
            Defaults: 1024.
        output_dim (int): The output feature dimension of FFNs.
            Default: None. If None, the `embed_dim` will be used.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        activation (nn.Module): The activation layer used in FFNs.
            Default: nn.ReLU(inplace=True).
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
    """

    def __init__(
        self,
        k_groups, 
        embed_dim,
        feedforward_dim,
        output_dim,
        num_fcs,
        activation=nn.ReLU(inplace=True),
        fc_bias=True,
        add_identity=True,
        ffn_drop=0.0,
        weight_predictor_type='linear'):
        super(DynamicFFN, self).__init__()
        assert num_fcs >= 2, "num_fcs should be no less " f"than 2. got {num_fcs}."
        self.embed_dim = embed_dim
        self.feedforward_dim = feedforward_dim
        self.num_fcs = num_fcs
        self.activation = activation

        output_dim = embed_dim if output_dim is None else output_dim

        layers = nn.ModuleList()
        in_channels = embed_dim
        for _ in range(num_fcs - 1):
            layers.append(
                CustomBlock(in_channels, feedforward_dim, bias=fc_bias, ffn_drop=ffn_drop, activation=self.activation, k_groups=k_groups)
            )
            in_channels = feedforward_dim
            
        layers.append(CustomLinear(in_channels, output_dim, bias=fc_bias, k_groups=k_groups))

        self.add_identity = add_identity
        self.layers = layers
    
            
    def forward(self, x, identity=None, **kwargs) -> torch.Tensor:
        """Forward function of `FFN`.

        Args:
            x (torch.Tensor): the input tensor used in `FFN` layers.
            identity (torch.Tensor): the tensor with the same shape as `x`,
                which will be used for identity addition. Default: None.
                if None, `x` will be used.

        Returns:
            torch.Tensor: the forward results of `FFN` layer
        """
        out = x
        for i in range(len(self.layers)):
            out = self.layers[i](out, **kwargs)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out



class CustomTransformerLayerSequence(nn.Module):
    """Base class for TransformerEncoder and TransformerDecoder, which will copy
    the passed `transformer_layers` module `num_layers` time or save the passed
    list of `transformer_layers` as parameters named ``self.layers``
    which is the type of ``nn.ModuleList``.
    The users should inherit `TransformerLayerSequence` and implemente their
    own forward function.

    Args:
        transformer_layers (list[BaseTransformerLayer] | BaseTransformerLayer): A list
            of BaseTransformerLayer. If it is obj:`BaseTransformerLayer`, it
            would be repeated `num_layers` times to a list[BaseTransformerLayer]
        num_layers (int): The number of `TransformerLayer`. Default: None.
    """

    def __init__(
        self,
        transformer_layers=None,
        num_layers=None,
        transformer_layers_custom=None,
        custom_layers_index=None
    ):
        super(CustomTransformerLayerSequence, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        if isinstance(transformer_layers, nn.Module):
            for k in range(num_layers):
                if k in custom_layers_index:
                    self.layers.append(copy.deepcopy(transformer_layers_custom))
                else:
                    self.layers.append(copy.deepcopy(transformer_layers))
        else:
            assert isinstance(transformer_layers, list) and len(transformer_layers) == num_layers

    def forward(self):
        """Forward function of `TransformerLayerSequence`. The users should inherit
        `TransformerLayerSequence` and implemente their own forward function.
        """
        raise NotImplementedError()