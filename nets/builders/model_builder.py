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

from nets.builders import anchor_generator_builder
from nets.builders import box_coder_builder
from nets.builders import box_predictor_builder
from nets.builders import hyperparams_builder
from nets.builders import image_resizer_builder
from nets.builders import losses_builder
from nets.builders import matcher_builder
from nets.builders import post_processing_builder
from nets.builders import region_similarity_calculator_builder as sim_calc
from nets.core import target_assigner
from nets.meta_architectures import ssd_meta_arch
from nets.models import ssd_efficientnet_bifpn_feature_extractor as ssd_efficientnet_bifpn
from nets.protos import model_pb2
from nets.utils import ops

SSD_KERAS_FEATURE_EXTRACTOR_CLASS_MAP = {
    'ssd_efficientnet-b0_bifpn_keras': ssd_efficientnet_bifpn.SSDEfficientNetB0BiFPNKerasFeatureExtractor,
    'ssd_efficientnet-b1_bifpn_keras': ssd_efficientnet_bifpn.SSDEfficientNetB1BiFPNKerasFeatureExtractor,
    'ssd_efficientnet-b2_bifpn_keras': ssd_efficientnet_bifpn.SSDEfficientNetB2BiFPNKerasFeatureExtractor,
    'ssd_efficientnet-b3_bifpn_keras': ssd_efficientnet_bifpn.SSDEfficientNetB3BiFPNKerasFeatureExtractor,
    'ssd_efficientnet-b4_bifpn_keras': ssd_efficientnet_bifpn.SSDEfficientNetB4BiFPNKerasFeatureExtractor,
    'ssd_efficientnet-b5_bifpn_keras': ssd_efficientnet_bifpn.SSDEfficientNetB5BiFPNKerasFeatureExtractor,
    'ssd_efficientnet-b6_bifpn_keras': ssd_efficientnet_bifpn.SSDEfficientNetB6BiFPNKerasFeatureExtractor,
    'ssd_efficientnet-b7_bifpn_keras': ssd_efficientnet_bifpn.SSDEfficientNetB7BiFPNKerasFeatureExtractor, }

FEATURE_EXTRACTOR_MAPS = [SSD_KERAS_FEATURE_EXTRACTOR_CLASS_MAP]


def _check_feature_extractor_exists(feature_extractor_type):
    feature_extractors = set().union(*FEATURE_EXTRACTOR_MAPS)
    if feature_extractor_type not in feature_extractors:
        raise ValueError('{} is not supported. See `model_builder.py` for features '
                         'extractors compatible with different versions of '
                         'Tensorflow'.format(feature_extractor_type))


def _build_ssd_feature_extractor(feature_extractor_config,
                                 is_training,
                                 freeze_batchnorm,
                                 reuse_weights=None):
    feature_type = feature_extractor_config.type
    depth_multiplier = feature_extractor_config.depth_multiplier
    min_depth = feature_extractor_config.min_depth
    pad_to_multiple = feature_extractor_config.pad_to_multiple
    use_explicit_padding = feature_extractor_config.use_explicit_padding
    use_depthwise = feature_extractor_config.use_depthwise

    conv_hyperparams = hyperparams_builder.KerasLayerHyperparams(feature_extractor_config.conv_hyperparams)

    override_base_feature_extractor_hyperparams = (feature_extractor_config.override_base_feature_extractor_hyperparams)

    feature_extractor_class = SSD_KERAS_FEATURE_EXTRACTOR_CLASS_MAP[feature_type]

    kwargs = {'is_training': is_training,
              'depth_multiplier': depth_multiplier,
              'min_depth': min_depth,
              'pad_to_multiple': pad_to_multiple,
              'use_explicit_padding': use_explicit_padding,
              'use_depthwise': use_depthwise,
              'override_base_feature_extractor_hyperparams': override_base_feature_extractor_hyperparams}

    if feature_extractor_config.HasField('replace_preprocessor_with_placeholder'):
        kwargs.update(
            {'replace_preprocessor_with_placeholder': feature_extractor_config.replace_preprocessor_with_placeholder})

    if feature_extractor_config.HasField('num_layers'):
        kwargs.update({'num_layers': feature_extractor_config.num_layers})

    kwargs.update({'conv_hyperparams': conv_hyperparams,
                   'inplace_batchnorm_update': False,
                   'freeze_batchnorm': freeze_batchnorm})

    if feature_extractor_config.HasField('fpn'):
        kwargs.update({'fpn_min_level': feature_extractor_config.fpn.min_level,
                       'fpn_max_level': feature_extractor_config.fpn.max_level,
                       'additional_layer_depth': feature_extractor_config.fpn.additional_layer_depth, })

    if feature_extractor_config.HasField('bifpn'):
        kwargs.update({'bifpn_min_level': feature_extractor_config.bifpn.min_level,
                       'bifpn_max_level': feature_extractor_config.bifpn.max_level,
                       'bifpn_num_iterations': feature_extractor_config.bifpn.num_iterations,
                       'bifpn_num_filters': feature_extractor_config.bifpn.num_filters,
                       'bifpn_combine_method': feature_extractor_config.bifpn.combine_method, })

    return feature_extractor_class(**kwargs)


def _build_ssd_model(ssd_config, is_training, add_summaries):
    num_classes = ssd_config.num_classes
    _check_feature_extractor_exists(ssd_config.feature_extractor.type)

    feature_extractor = _build_ssd_feature_extractor(feature_extractor_config=ssd_config.feature_extractor,
                                                     freeze_batchnorm=ssd_config.freeze_batchnorm,
                                                     is_training=is_training)

    box_coder = box_coder_builder.build(ssd_config.box_coder)
    matcher = matcher_builder.build(ssd_config.matcher)
    region_similarity_calculator = sim_calc.build(ssd_config.similarity_calculator)
    encode_background_as_zeros = ssd_config.encode_background_as_zeros
    negative_class_weight = ssd_config.negative_class_weight
    anchor_generator = anchor_generator_builder.build(ssd_config.anchor_generator)

    ssd_box_predictor = box_predictor_builder.build_keras(hyperparams_builder.KerasLayerHyperparams,
                                                          ssd_config.freeze_batchnorm,
                                                          False,
                                                          anchor_generator.num_anchors_per_location(),
                                                          ssd_config.box_predictor,
                                                          is_training,
                                                          num_classes,
                                                          ssd_config.add_background_class)

    image_resizer_fn = image_resizer_builder.build(ssd_config.image_resizer)
    non_max_suppression_fn, score_conversion_fn = post_processing_builder.build(ssd_config.post_processing)
    (classification_loss, localization_loss, classification_weight,
     localization_weight, hard_example_miner, random_example_sampler,
     expected_loss_weights_fn) = losses_builder.build(ssd_config.loss)
    normalize_loss_by_num_matches = ssd_config.normalize_loss_by_num_matches
    normalize_loc_loss_by_codesize = ssd_config.normalize_loc_loss_by_codesize

    equalization_loss_config = ops.EqualizationLossConfig(ssd_config.loss.equalization_loss.weight,
                                                          ssd_config.loss.equalization_loss.exclude_prefixes)

    target_assigner_instance = target_assigner.TargetAssigner(region_similarity_calculator,
                                                              matcher,
                                                              box_coder,
                                                              negative_class_weight)

    ssd_meta_arch_fn = ssd_meta_arch.SSDMetaArch
    kwargs = {}

    return ssd_meta_arch_fn(is_training=is_training,
                            anchor_generator=anchor_generator,
                            box_predictor=ssd_box_predictor,
                            box_coder=box_coder,
                            feature_extractor=feature_extractor,
                            encode_background_as_zeros=encode_background_as_zeros,
                            image_resizer_fn=image_resizer_fn,
                            non_max_suppression_fn=non_max_suppression_fn,
                            score_conversion_fn=score_conversion_fn,
                            classification_loss=classification_loss,
                            localization_loss=localization_loss,
                            classification_loss_weight=classification_weight,
                            localization_loss_weight=localization_weight,
                            normalize_loss_by_num_matches=normalize_loss_by_num_matches,
                            hard_example_miner=hard_example_miner,
                            target_assigner_instance=target_assigner_instance,
                            add_summaries=add_summaries,
                            normalize_loc_loss_by_codesize=normalize_loc_loss_by_codesize,
                            freeze_batchnorm=ssd_config.freeze_batchnorm,
                            inplace_batchnorm_update=ssd_config.inplace_batchnorm_update,
                            add_background_class=ssd_config.add_background_class,
                            explicit_background_class=ssd_config.explicit_background_class,
                            random_example_sampler=random_example_sampler,
                            expected_loss_weights_fn=expected_loss_weights_fn,
                            use_confidences_as_targets=ssd_config.use_confidences_as_targets,
                            implicit_example_weight=ssd_config.implicit_example_weight,
                            equalization_loss_config=equalization_loss_config,
                            return_raw_detections_during_predict=ssd_config.return_raw_detections_during_predict,
                            **kwargs)


EXPERIMENTAL_META_ARCH_BUILDER_MAP = {}

META_ARCH_BUILDER_MAP = {'ssd': _build_ssd_model}


def build(model_config, is_training, add_summaries=True):
    if not isinstance(model_config, model_pb2.DetectionModel):
        raise ValueError('model_config not of type model_pb2.DetectionModel.')

    meta_architecture = model_config.WhichOneof('model')

    if meta_architecture not in META_ARCH_BUILDER_MAP:
        raise ValueError('Unknown meta architecture: {}'.format(meta_architecture))
    else:
        build_func = META_ARCH_BUILDER_MAP[meta_architecture]
        return build_func(getattr(model_config, meta_architecture), is_training, add_summaries)
