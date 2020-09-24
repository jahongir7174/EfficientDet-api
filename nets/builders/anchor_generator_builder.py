# Lint as: python2, python3
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""A function to build an object detection anchor generator from config."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from nets.anchor_generators import multiscale_grid_anchor_generator
from nets.protos import anchor_generator_pb2


def build(anchor_generator_config):
    """Builds an anchor generator based on the config.

    Args:
      anchor_generator_config: An anchor_generator.proto object containing the
        config for the desired anchor generator.

    Returns:
      Anchor generator based on the config.

    Raises:
      ValueError: On empty anchor generator proto.
    """
    if not isinstance(anchor_generator_config, anchor_generator_pb2.AnchorGenerator):
        raise ValueError('anchor_generator_config not of type anchor_generator_pb2.AnchorGenerator')
    if anchor_generator_config.WhichOneof('anchor_generator_oneof') == 'multiscale_anchor_generator':
        cfg = anchor_generator_config.multiscale_anchor_generator
        return multiscale_grid_anchor_generator.MultiscaleGridAnchorGenerator(
            cfg.min_level,
            cfg.max_level,
            cfg.anchor_scale,
            [float(aspect_ratio) for aspect_ratio in cfg.aspect_ratios],
            cfg.scales_per_octave,
            cfg.normalize_coordinates
        )
    else:
        raise ValueError('Empty anchor generator.')
