# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow.compat.v1 as tf

from nets import inputs
from nets.builders import model_builder
from nets.builders import optimizer_builder
from nets.core import standard_fields as fields
from nets.protos import train_pb2
from nets.utils import config_util
from nets.utils import ops
from nets.utils import shape_utils

RESTORE_MAP_ERROR_TEMPLATE = ('Since we are restoring a v2 style checkpoint'
                              ' restore_map was expected to return a (str -> Model) mapping,'
                              ' but we received a ({} -> {}) mapping instead.')

MODEL_BUILD_UTIL_MAP = {'get_configs_from_pipeline_file': config_util.get_configs_from_pipeline_file,
                        'create_pipeline_proto_from_configs': config_util.create_pipeline_proto_from_configs,
                        'merge_external_params_with_configs': config_util.merge_external_params_with_configs,
                        'create_train_input_fn': inputs.create_train_input_fn,
                        'create_eval_input_fn': inputs.create_eval_input_fn,
                        'create_predict_input_fn': inputs.create_predict_input_fn,
                        'detection_model_fn_base': model_builder.build, }


def _prepare_groundtruth_for_eval(detection_model, class_agnostic, max_number_of_boxes):
    input_data_fields = fields.InputDataFields()
    groundtruth_boxes = tf.stack(detection_model.groundtruth_lists(fields.BoxListFields.boxes))
    groundtruth_boxes_shape = tf.shape(groundtruth_boxes)
    if class_agnostic:
        groundtruth_classes_one_hot = tf.ones([groundtruth_boxes_shape[0], groundtruth_boxes_shape[1], 1])
    else:
        groundtruth_classes_one_hot = tf.stack(detection_model.groundtruth_lists(fields.BoxListFields.classes))
    label_id_offset = 1
    groundtruth_classes = (tf.argmax(groundtruth_classes_one_hot, axis=2) + label_id_offset)
    groundtruth = {input_data_fields.groundtruth_boxes: groundtruth_boxes,
                   input_data_fields.groundtruth_classes: groundtruth_classes}
    if detection_model.groundtruth_has_field(fields.BoxListFields.masks):
        groundtruth[input_data_fields.groundtruth_instance_masks] = tf.stack(
            detection_model.groundtruth_lists(fields.BoxListFields.masks))

    if detection_model.groundtruth_has_field(fields.BoxListFields.is_crowd):
        groundtruth[input_data_fields.groundtruth_is_crowd] = tf.stack(
            detection_model.groundtruth_lists(fields.BoxListFields.is_crowd))

    if detection_model.groundtruth_has_field(input_data_fields.groundtruth_area):
        groundtruth[input_data_fields.groundtruth_area] = tf.stack(
            detection_model.groundtruth_lists(input_data_fields.groundtruth_area))

    if detection_model.groundtruth_has_field(fields.BoxListFields.keypoints):
        groundtruth[input_data_fields.groundtruth_keypoints] = tf.stack(
            detection_model.groundtruth_lists(fields.BoxListFields.keypoints))

    if detection_model.groundtruth_has_field(fields.BoxListFields.keypoint_visibilities):
        groundtruth[input_data_fields.groundtruth_keypoint_visibilities] = tf.stack(
            detection_model.groundtruth_lists(fields.BoxListFields.keypoint_visibilities))

    if detection_model.groundtruth_has_field(fields.BoxListFields.group_of):
        groundtruth[input_data_fields.groundtruth_group_of] = tf.stack(
            detection_model.groundtruth_lists(fields.BoxListFields.group_of))

    if detection_model.groundtruth_has_field(
            fields.InputDataFields.groundtruth_labeled_classes):
        labeled_classes_list = detection_model.groundtruth_lists(
            fields.InputDataFields.groundtruth_labeled_classes)
        labeled_classes = [tf.where(x)[:, 0] + label_id_offset for x in labeled_classes_list]
        if len(labeled_classes) > 1:
            num_classes = labeled_classes_list[0].shape[0]
            padded_labeled_classes = []
            for x in labeled_classes:
                padding = num_classes - tf.shape(x)[0]
                padded_labeled_classes.append(tf.pad(x, [[0, padding]]))
            groundtruth[input_data_fields.groundtruth_labeled_classes] = tf.stack(
                padded_labeled_classes)
        else:
            groundtruth[input_data_fields.groundtruth_labeled_classes] = tf.stack(labeled_classes)

    if detection_model.groundtruth_has_field(fields.BoxListFields.densepose_num_points):
        groundtruth[input_data_fields.groundtruth_dp_num_points] = tf.stack(
            detection_model.groundtruth_lists(fields.BoxListFields.densepose_num_points))
    if detection_model.groundtruth_has_field(
            fields.BoxListFields.densepose_part_ids):
        groundtruth[input_data_fields.groundtruth_dp_part_ids] = tf.stack(
            detection_model.groundtruth_lists(
                fields.BoxListFields.densepose_part_ids))
    if detection_model.groundtruth_has_field(
            fields.BoxListFields.densepose_surface_coords):
        groundtruth[input_data_fields.groundtruth_dp_surface_coords] = tf.stack(
            detection_model.groundtruth_lists(
                fields.BoxListFields.densepose_surface_coords))

    if detection_model.groundtruth_has_field(fields.BoxListFields.track_ids):
        groundtruth[input_data_fields.groundtruth_track_ids] = tf.stack(
            detection_model.groundtruth_lists(fields.BoxListFields.track_ids))

    groundtruth[input_data_fields.num_groundtruth_boxes] = (
        tf.tile([max_number_of_boxes], multiples=[groundtruth_boxes_shape[0]]))
    return groundtruth


def unstack_batch(tensor_dict, unpad_groundtruth_tensors=True):
    unbatched_tensor_dict = {key: tf.unstack(tensor) for key, tensor in tensor_dict.items()}
    if unpad_groundtruth_tensors:
        if (fields.InputDataFields.num_groundtruth_boxes not in
                unbatched_tensor_dict):
            raise ValueError('`num_groundtruth_boxes` not found in tensor_dict. '
                             'Keys available: {}'.format(unbatched_tensor_dict.keys()))
        unbatched_unpadded_tensor_dict = {}
        unpad_keys = {fields.InputDataFields.groundtruth_instance_masks,
                      fields.InputDataFields.groundtruth_classes,
                      fields.InputDataFields.groundtruth_boxes,
                      fields.InputDataFields.groundtruth_keypoints,
                      fields.InputDataFields.groundtruth_keypoint_visibilities,
                      fields.InputDataFields.groundtruth_dp_num_points,
                      fields.InputDataFields.groundtruth_dp_part_ids,
                      fields.InputDataFields.groundtruth_dp_surface_coords,
                      fields.InputDataFields.groundtruth_track_ids,
                      fields.InputDataFields.groundtruth_group_of,
                      fields.InputDataFields.groundtruth_difficult,
                      fields.InputDataFields.groundtruth_is_crowd,
                      fields.InputDataFields.groundtruth_area,
                      fields.InputDataFields.groundtruth_weights}.intersection(set(unbatched_tensor_dict.keys()))

        for key in unpad_keys:
            unpadded_tensor_list = []
            for num_gt, padded_tensor in zip(
                    unbatched_tensor_dict[fields.InputDataFields.num_groundtruth_boxes],
                    unbatched_tensor_dict[key]):
                tensor_shape = shape_utils.combined_static_and_dynamic_shape(
                    padded_tensor)
                slice_begin = tf.zeros([len(tensor_shape)], dtype=tf.int32)
                slice_size = tf.stack(
                    [num_gt] + [-1 if dim is None else dim for dim in tensor_shape[1:]])
                unpadded_tensor = tf.slice(padded_tensor, slice_begin, slice_size)
                unpadded_tensor_list.append(unpadded_tensor)
            unbatched_unpadded_tensor_dict[key] = unpadded_tensor_list

        unbatched_tensor_dict.update(unbatched_unpadded_tensor_dict)

    return unbatched_tensor_dict


def provide_groundtruth(model, labels):
    gt_boxes_list = labels[fields.InputDataFields.groundtruth_boxes]
    gt_classes_list = labels[fields.InputDataFields.groundtruth_classes]
    gt_masks_list = None
    if fields.InputDataFields.groundtruth_instance_masks in labels:
        gt_masks_list = labels[fields.InputDataFields.groundtruth_instance_masks]
    gt_keypoints_list = None
    if fields.InputDataFields.groundtruth_keypoints in labels:
        gt_keypoints_list = labels[fields.InputDataFields.groundtruth_keypoints]
    gt_keypoint_visibilities_list = None
    if fields.InputDataFields.groundtruth_keypoint_visibilities in labels:
        gt_keypoint_visibilities_list = labels[fields.InputDataFields.groundtruth_keypoint_visibilities]
    gt_dp_num_points_list = None
    if fields.InputDataFields.groundtruth_dp_num_points in labels:
        gt_dp_num_points_list = labels[fields.InputDataFields.groundtruth_dp_num_points]
    gt_dp_part_ids_list = None
    if fields.InputDataFields.groundtruth_dp_part_ids in labels:
        gt_dp_part_ids_list = labels[fields.InputDataFields.groundtruth_dp_part_ids]
    gt_dp_surface_coords_list = None
    if fields.InputDataFields.groundtruth_dp_surface_coords in labels:
        gt_dp_surface_coords_list = labels[fields.InputDataFields.groundtruth_dp_surface_coords]
    gt_track_ids_list = None
    if fields.InputDataFields.groundtruth_track_ids in labels:
        gt_track_ids_list = labels[fields.InputDataFields.groundtruth_track_ids]
    gt_weights_list = None
    if fields.InputDataFields.groundtruth_weights in labels:
        gt_weights_list = labels[fields.InputDataFields.groundtruth_weights]
    gt_confidences_list = None
    if fields.InputDataFields.groundtruth_confidences in labels:
        gt_confidences_list = labels[fields.InputDataFields.groundtruth_confidences]
    gt_is_crowd_list = None
    if fields.InputDataFields.groundtruth_is_crowd in labels:
        gt_is_crowd_list = labels[fields.InputDataFields.groundtruth_is_crowd]
    gt_group_of_list = None
    if fields.InputDataFields.groundtruth_group_of in labels:
        gt_group_of_list = labels[fields.InputDataFields.groundtruth_group_of]
    gt_area_list = None
    if fields.InputDataFields.groundtruth_area in labels:
        gt_area_list = labels[fields.InputDataFields.groundtruth_area]
    gt_labeled_classes = None
    if fields.InputDataFields.groundtruth_labeled_classes in labels:
        gt_labeled_classes = labels[fields.InputDataFields.groundtruth_labeled_classes]
    model.provide_groundtruth(groundtruth_boxes_list=gt_boxes_list,
                              groundtruth_classes_list=gt_classes_list,
                              groundtruth_confidences_list=gt_confidences_list,
                              groundtruth_labeled_classes=gt_labeled_classes,
                              groundtruth_masks_list=gt_masks_list,
                              groundtruth_keypoints_list=gt_keypoints_list,
                              groundtruth_keypoint_visibilities_list=gt_keypoint_visibilities_list,
                              groundtruth_dp_num_points_list=gt_dp_num_points_list,
                              groundtruth_dp_part_ids_list=gt_dp_part_ids_list,
                              groundtruth_dp_surface_coords_list=gt_dp_surface_coords_list,
                              groundtruth_weights_list=gt_weights_list,
                              groundtruth_is_crowd_list=gt_is_crowd_list,
                              groundtruth_group_of_list=gt_group_of_list,
                              groundtruth_area_list=gt_area_list,
                              groundtruth_track_ids_list=gt_track_ids_list)


def _compute_losses_and_predictions_dicts(model, features, labels, add_regularization_loss=True):
    provide_groundtruth(model, labels)
    preprocessed_images = features[fields.InputDataFields.image]

    prediction_dict = model.predict(preprocessed_images, features[fields.InputDataFields.true_image_shape], )
    prediction_dict = ops.bfloat16_to_float32_nested(prediction_dict)

    losses_dict = model.loss(prediction_dict, features[fields.InputDataFields.true_image_shape])
    losses = [loss_tensor for loss_tensor in losses_dict.values()]
    if add_regularization_loss:
        regularization_losses = model.regularization_losses()
        if regularization_losses:
            regularization_losses = ops.bfloat16_to_float32_nested(regularization_losses)
            regularization_loss = tf.add_n(regularization_losses, name='regularization_loss')
            losses.append(regularization_loss)
            losses_dict['Loss/regularization_loss'] = regularization_loss

    total_loss = tf.add_n(losses, name='total_loss')
    losses_dict['Loss/total_loss'] = total_loss

    return losses_dict, prediction_dict


def eager_train_step(detection_model,
                     features,
                     labels,
                     unpad_groundtruth_tensors,
                     optimizer,
                     learning_rate,
                     add_regularization_loss=True,
                     clip_gradients_value=None,
                     global_step=None,
                     num_replicas=1.0):
    is_training = True

    detection_model._is_training = is_training
    tf.keras.backend.set_learning_phase(is_training)

    labels = unstack_batch(labels, unpad_groundtruth_tensors=unpad_groundtruth_tensors)

    with tf.GradientTape() as tape:
        losses_dict, _ = _compute_losses_and_predictions_dicts(detection_model,
                                                               features, labels,
                                                               add_regularization_loss)

        total_loss = losses_dict['Loss/total_loss']

        total_loss = tf.math.divide(total_loss, tf.constant(num_replicas, dtype=tf.float32))
        losses_dict['Loss/normalized_total_loss'] = total_loss

    for loss_type in losses_dict:
        tf.compat.v2.summary.scalar(loss_type, losses_dict[loss_type], step=global_step)

    trainable_variables = detection_model.trainable_variables

    gradients = tape.gradient(total_loss, trainable_variables)

    if clip_gradients_value:
        gradients, _ = tf.clip_by_global_norm(gradients, clip_gradients_value)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    tf.compat.v2.summary.scalar('learning_rate', learning_rate, step=global_step)
    tf.compat.v2.summary.image(name='train_input_images',
                               step=global_step,
                               data=features[fields.InputDataFields.image],
                               max_outputs=3)
    return total_loss


def validate_tf_v2_checkpoint_restore_map(checkpoint_restore_map):
    for key, value in checkpoint_restore_map.items():
        if not (isinstance(key, str) and
                (isinstance(value, tf.Module)
                 or isinstance(value, tf.train.Checkpoint))):
            if isinstance(key, str) and isinstance(value, dict):
                validate_tf_v2_checkpoint_restore_map(value)
            else:
                raise TypeError(
                    RESTORE_MAP_ERROR_TEMPLATE.format(key.__class__.__name__, value.__class__.__name__))


def is_object_based_checkpoint(checkpoint_path):
    var_names = [var[0] for var in tf.train.list_variables(checkpoint_path)]
    return '_CHECKPOINTABLE_OBJECT_GRAPH' in var_names


def load_fine_tune_checkpoint(model,
                              checkpoint_path,
                              checkpoint_type,
                              checkpoint_version,
                              input_dataset,
                              unpad_groundtruth_tensors):
    if not is_object_based_checkpoint(checkpoint_path):
        raise IOError('Checkpoint is expected to be an object-based checkpoint.')
    if checkpoint_version == train_pb2.CheckpointVersion.V1:
        raise ValueError('Checkpoint version should be V2')

    features, labels = iter(input_dataset).next()

    @tf.function
    def _dummy_computation_fn(features, labels):
        model._is_training = False
        tf.keras.backend.set_learning_phase(False)

        labels = unstack_batch(labels, unpad_groundtruth_tensors=unpad_groundtruth_tensors)

        return _compute_losses_and_predictions_dicts(model,
                                                     features,
                                                     labels)

    strategy = tf.compat.v2.distribute.get_strategy()
    if hasattr(tf.distribute.Strategy, 'run'):
        strategy.run(_dummy_computation_fn, args=(features, labels,))
    else:
        strategy.experimental_run_v2(_dummy_computation_fn, args=(features, labels,))

    restore_from_objects_dict = model.restore_from_objects(fine_tune_checkpoint_type=checkpoint_type)
    validate_tf_v2_checkpoint_restore_map(restore_from_objects_dict)
    ckpt = tf.train.Checkpoint(**restore_from_objects_dict)
    ckpt.restore(checkpoint_path).assert_existing_objects_matched()


def get_filepath(strategy, filepath):
    if strategy.extended.should_checkpoint:
        return filepath
    else:
        task_id = strategy.extended._task_id
        return os.path.join(filepath, 'temp_worker_{:03d}'.format(task_id))


def clean_temporary_directories(strategy, filepath):
    if not strategy.extended.should_checkpoint:
        if tf.io.gfile.exists(filepath) and tf.io.gfile.isdir(filepath):
            tf.io.gfile.rmtree(filepath)


def train_loop(pipeline_config_path,
               model_dir,
               config_override=None,
               train_steps=None,
               use_tpu=False,
               save_final_config=False,
               checkpoint_every_n=1000,
               checkpoint_max_to_keep=7,
               **kwargs):
    get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP['get_configs_from_pipeline_file']
    merge_external_params_with_configs = MODEL_BUILD_UTIL_MAP['merge_external_params_with_configs']
    create_pipeline_proto_from_configs = MODEL_BUILD_UTIL_MAP['create_pipeline_proto_from_configs']

    configs = get_configs_from_pipeline_file(pipeline_config_path, config_override=config_override)
    kwargs.update({'train_steps': train_steps, 'use_bfloat16': configs['train_config'].use_bfloat16 and use_tpu})
    configs = merge_external_params_with_configs(configs, None, kwargs_dict=kwargs)
    model_config = configs['model']
    train_config = configs['train_config']
    train_input_config = configs['train_input_config']

    unpad_groundtruth_tensors = train_config.unpad_groundtruth_tensors
    add_regularization_loss = train_config.add_regularization_loss
    clip_gradients_value = None
    if train_config.gradient_clipping_by_norm > 0:
        clip_gradients_value = train_config.gradient_clipping_by_norm

    if train_steps is None and train_config.num_steps != 0:
        train_steps = train_config.num_steps

    if kwargs['use_bfloat16']:
        tf.compat.v2.keras.mixed_precision.experimental.set_policy('mixed_bfloat16')

    if train_config.load_all_detection_checkpoint_vars:
        raise ValueError('train_pb2.load_all_detection_checkpoint_vars unsupported in TF2')

    config_util.update_fine_tune_checkpoint_type(train_config)
    fine_tune_checkpoint_type = train_config.fine_tune_checkpoint_type
    fine_tune_checkpoint_version = train_config.fine_tune_checkpoint_version

    if save_final_config:
        pipeline_config_final = create_pipeline_proto_from_configs(configs)
        config_util.save_pipeline_config(pipeline_config_final, model_dir)

    strategy = tf.compat.v2.distribute.get_strategy()
    with strategy.scope():
        detection_model = MODEL_BUILD_UTIL_MAP['detection_model_fn_base'](model_config=model_config, is_training=True)

        def train_dataset_fn(input_context):
            train_input = inputs.train_input(train_config=train_config,
                                             train_input_config=train_input_config,
                                             model_config=model_config,
                                             model=detection_model,
                                             input_context=input_context)
            train_input = train_input.repeat()
            return train_input

        train_input = strategy.experimental_distribute_datasets_from_function(train_dataset_fn)

        global_step = tf.Variable(0, trainable=False, dtype=tf.compat.v2.dtypes.int64, name='global_step',
                                  aggregation=tf.compat.v2.VariableAggregation.ONLY_FIRST_REPLICA)
        optimizer, (learning_rate,) = optimizer_builder.build(train_config.optimizer, global_step=global_step)

        if callable(learning_rate):
            learning_rate_fn = learning_rate
        else:
            learning_rate_fn = lambda: learning_rate

    summary_writer_filepath = get_filepath(strategy, os.path.join(model_dir, 'train'))

    if use_tpu:
        num_steps_per_iteration = 100
    else:
        num_steps_per_iteration = 1

    with strategy.scope():
        with tf.compat.v2.summary.record_if(lambda: global_step % num_steps_per_iteration == 0):
            # Load a fine-tuning checkpoint.
            if train_config.fine_tune_checkpoint:
                load_fine_tune_checkpoint(detection_model,
                                          train_config.fine_tune_checkpoint,
                                          fine_tune_checkpoint_type,
                                          fine_tune_checkpoint_version,
                                          train_input,
                                          unpad_groundtruth_tensors)

            ckpt = tf.compat.v2.train.Checkpoint(step=global_step, model=detection_model, optimizer=optimizer)

            manager_dir = get_filepath(strategy, model_dir)
            if not strategy.extended.should_checkpoint:
                checkpoint_max_to_keep = 1
            manager = tf.compat.v2.train.CheckpointManager(ckpt, manager_dir, max_to_keep=checkpoint_max_to_keep)

            latest_checkpoint = tf.train.latest_checkpoint(model_dir)
            ckpt.restore(latest_checkpoint)

            def train_step_fn(features, labels):
                loss = eager_train_step(detection_model,
                                        features,
                                        labels,
                                        unpad_groundtruth_tensors,
                                        optimizer,
                                        learning_rate=learning_rate_fn(),
                                        add_regularization_loss=add_regularization_loss,
                                        clip_gradients_value=clip_gradients_value,
                                        global_step=global_step,
                                        num_replicas=strategy.num_replicas_in_sync)
                global_step.assign_add(1)
                return loss

            def _sample_and_train(strategy, train_step_fn, data_iterator):
                features, labels = data_iterator.next()
                if hasattr(tf.distribute.Strategy, 'run'):
                    per_replica_losses = strategy.run(train_step_fn, args=(features, labels))
                else:
                    per_replica_losses = strategy.experimental_run_v2(train_step_fn, args=(features, labels))
                return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

            @tf.function
            def _dist_train_step(data_iterator):

                if num_steps_per_iteration > 1:
                    for _ in tf.range(num_steps_per_iteration - 1):
                        with tf.name_scope(''):
                            _sample_and_train(strategy, train_step_fn, data_iterator)

                return _sample_and_train(strategy, train_step_fn, data_iterator)

            train_input_iter = iter(train_input)

            checkpoint_step = int(global_step.value())
            logged_step = global_step.value()

            last_step_time = time.time()
            for _ in range(global_step.value(), train_steps,
                           num_steps_per_iteration):

                loss = _dist_train_step(train_input_iter)

                time_taken = time.time() - last_step_time
                last_step_time = time.time()

                tf.compat.v2.summary.scalar('steps_per_sec',
                                            num_steps_per_iteration * 1.0 / time_taken,
                                            step=global_step)

                if global_step.value() - logged_step >= 1:
                    ps_time = time_taken / num_steps_per_iteration
                    print(f'Step {global_step.value()} per-step time {ps_time:.3f}s loss={loss:.3f}')
                    logged_step = global_step.value()

                if (int(global_step.value()) - checkpoint_step) >= checkpoint_every_n:
                    manager.save()
                    checkpoint_step = int(global_step.value())

    clean_temporary_directories(strategy, manager_dir)
    clean_temporary_directories(strategy, summary_writer_filepath)
