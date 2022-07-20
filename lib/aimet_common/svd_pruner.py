# /usr/bin/env python3.5
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

""" Prunes layers using Spatial SVD or Weight SVD schemes """

import abc
from typing import Tuple

import numpy as np

from lib.aimet_common.utils import AimetLogger
from lib.aimet_common.defs import CostMetric
from lib.aimet_common import cost_calculator
from lib.aimet_common.pruner import Pruner
from lib.aimet_common.layer_database import LayerDatabase, Layer

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Svd)


class SpatialSvdPruner(Pruner):
    """
    Pruner for Spatial-SVD method
    """

    def _prune_layer(self, orig_layer_db: LayerDatabase, comp_layer_db: LayerDatabase, layer: Layer,
                     comp_ratio: float, cost_metric: CostMetric):

        # Given a compression ratio, find the appropriate rank
        rank = cost_calculator.SpatialSvdCostCalculator.calculate_rank_given_comp_ratio(layer, comp_ratio, cost_metric)

        logger.info("Spatial SVD splitting layer: %s using rank: %s", layer.name, rank)

        # For the rounded rank compute the new compression ratio
        actual_comp_ratio = cost_calculator.SpatialSvdCostCalculator.calculate_comp_ratio_given_rank(layer, rank,
                                                                                                     cost_metric)

        # Perform svd and split the layers
        self._perform_svd_and_split_layer(layer, rank, comp_layer_db)

        return actual_comp_ratio

    @abc.abstractmethod
    def _perform_svd_and_split_layer(self, layer: Layer, rank: int, comp_layer_db: LayerDatabase):
        """
        Performs spatial svd and splits given layer into two layers
        :param layer: Layer to split
        :param rank: Rank to use for spatial svd splitting
        :param comp_layer_db: Compressed layer db to update with the split layers
        :return: None
        """

    @staticmethod
    def lingalg_spatial_svd(weight_tensor: np.array, rank: int,
                            in_channels: int, out_channels: int, height: int, width: int) -> Tuple[np.array, np.array]:
        """
        Splits a weight tensor using spatial svd
        :param weight_tensor: Weight tensor in numpy format (shape: out_chan, in_chan, height, width)
        :param rank: Rank to use for svd split
        :param in_channels: Number of in-channels
        :param out_channels: Number of out-channels
        :param height: Kernel height
        :param width: Kernel width
        :return: Tuple of split tensors in numpy format (shape: out_chan, in_chan, height, width)
        """
        assert rank <= in_channels * height

        # Reshape into a 2D matrix - because that's what numpy needs
        weight_tensor = np.transpose(weight_tensor, [1, 2, 0, 3])  # in_channels height out_channels width
        weight_tensor = weight_tensor.reshape(in_channels * height, out_channels * width)

        v, s, h = np.linalg.svd(weight_tensor, full_matrices=False)

        v = v[:, :rank]
        s = s[:rank]
        h = h[:rank, :]
        sqrt_s = np.sqrt(s)
        v = v * sqrt_s
        h = sqrt_s.reshape(sqrt_s.shape[0], 1) * h
        # rank nw -> rank out_channels weight_tensor 1
        h = h.reshape([rank, out_channels, width, 1])
        # rank out_channels weight_tensor 1 -> out_channels rank 1 weight_tensor
        h = np.transpose(h, [1, 0, 3, 2])
        # ch rank -> in_channels 1 h rank  -> rank in_channels h 1
        v = v.reshape((in_channels, 1, height, rank))
        v = np.transpose(v, [3, 0, 2, 1])

        return h, v
