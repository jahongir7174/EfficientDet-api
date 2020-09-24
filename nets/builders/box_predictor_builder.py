import collections

import tensorflow.compat.v1 as tf

from nets.predictors import convolutional_keras_box_predictor
from nets.predictors.heads import keras_box_head
from nets.predictors.heads import keras_class_head
from nets.protos import box_predictor_pb2


def build_weight_shared_convolutional_keras_box_predictor(
        is_training,
        num_classes,
        conv_hyperparams,
        freeze_batchnorm,
        inplace_batchnorm_update,
        num_predictions_per_location_list,
        depth,
        num_layers_before_predictor,
        box_code_size,
        kernel_size=3,
        add_background_class=True,
        class_prediction_bias_init=0.0,
        use_dropout=False,
        dropout_keep_prob=0.8,
        share_prediction_tower=False,
        apply_batch_norm=True,
        use_depthwise=False,
        score_converter_fn=tf.identity,
        box_encodings_clip_range=None,
        name='WeightSharedConvolutionalBoxPredictor',
        keyword_args=None):
    """Builds the Keras WeightSharedConvolutionalBoxPredictor from the arguments.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
        containing hyperparameters for convolution ops.
      freeze_batchnorm: Whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      inplace_batchnorm_update: Whether to update batch norm moving average
        values inplace. When this is false train op must add a control
        dependency on tf.graphkeys.UPDATE_OPS collection in order to update
        batch norm statistics.
      num_predictions_per_location_list: A list of integers representing the
        number of box predictions to be made per spatial location for each
        feature map.
      depth: depth of conv layers.
      num_layers_before_predictor: Number of the additional conv layers before
        the predictor.
      box_code_size: Size of encoding for each box.
      kernel_size: Size of final convolution kernel.
      add_background_class: Whether to add an implicit background class.
      class_prediction_bias_init: constant value to initialize bias of the last
        conv2d layer before class prediction.
      use_dropout: Whether to apply dropout to class prediction head.
          dropout_keep_prob: Probability of keeping activiations.
      share_prediction_tower: Whether to share the multi-layer tower between box
        prediction and class prediction heads.
      apply_batch_norm: Whether to apply batch normalization to conv layers in
        this predictor.
      use_depthwise: Whether to use depthwise separable conv2d instead of conv2d.
      score_converter_fn: Callable score converter to perform elementwise op on
        class scores.
      box_encodings_clip_range: Min and max values for clipping the box_encodings.
      name: A string name scope to assign to the box predictor. If `None`, Keras
        will auto-generate one from the class name.
      keyword_args: A dictionary with additional args.

    Returns:
      A Keras WeightSharedConvolutionalBoxPredictor class.
    """
    if len(set(num_predictions_per_location_list)) > 1:
        raise ValueError('num predictions per location must be same for all'
                         'feature maps, found: {}'.format(
            num_predictions_per_location_list))
    num_predictions_per_location = num_predictions_per_location_list[0]

    box_prediction_head = keras_box_head.WeightSharedConvolutionalBoxHead(
        box_code_size=box_code_size,
        kernel_size=kernel_size,
        conv_hyperparams=conv_hyperparams,
        num_predictions_per_location=num_predictions_per_location,
        use_depthwise=use_depthwise,
        box_encodings_clip_range=box_encodings_clip_range,
        name='WeightSharedConvolutionalBoxHead')
    class_prediction_head = keras_class_head.WeightSharedConvolutionalClassHead(
        num_class_slots=(
            num_classes + 1 if add_background_class else num_classes),
        use_dropout=use_dropout,
        dropout_keep_prob=dropout_keep_prob,
        kernel_size=kernel_size,
        conv_hyperparams=conv_hyperparams,
        num_predictions_per_location=num_predictions_per_location,
        class_prediction_bias_init=class_prediction_bias_init,
        use_depthwise=use_depthwise,
        score_converter_fn=score_converter_fn,
        name='WeightSharedConvolutionalClassHead')
    other_heads = {}

    return (
        convolutional_keras_box_predictor.WeightSharedConvolutionalBoxPredictor(
            is_training=is_training,
            num_classes=num_classes,
            box_prediction_head=box_prediction_head,
            class_prediction_head=class_prediction_head,
            other_heads=other_heads,
            conv_hyperparams=conv_hyperparams,
            depth=depth,
            num_layers_before_predictor=num_layers_before_predictor,
            freeze_batchnorm=freeze_batchnorm,
            inplace_batchnorm_update=inplace_batchnorm_update,
            kernel_size=kernel_size,
            apply_batch_norm=apply_batch_norm,
            share_prediction_tower=share_prediction_tower,
            use_depthwise=use_depthwise,
            name=name))


def build_score_converter(score_converter_config, is_training):
    if score_converter_config == (box_predictor_pb2.WeightSharedConvolutionalBoxPredictor.IDENTITY):
        return tf.identity
    if score_converter_config == (box_predictor_pb2.WeightSharedConvolutionalBoxPredictor.SIGMOID):
        return tf.identity if is_training else tf.sigmoid
    raise ValueError('Unknown score converter.')


BoxEncodingsClipRange = collections.namedtuple('BoxEncodingsClipRange', ['min', 'max'])


def build_keras(hyperparams_fn,
                freeze_batchnorm,
                inplace_batchnorm_update,
                num_predictions_per_location_list,
                box_predictor_config,
                is_training,
                num_classes,
                add_background_class=True):
    if not isinstance(box_predictor_config, box_predictor_pb2.BoxPredictor):
        raise ValueError('box_predictor_config not of type box_predictor_pb2.BoxPredictor.')

    box_predictor_oneof = box_predictor_config.WhichOneof('box_predictor_oneof')

    if box_predictor_oneof == 'weight_shared_convolutional_box_predictor':
        config_box_predictor = (
            box_predictor_config.weight_shared_convolutional_box_predictor)
        conv_hyperparams = hyperparams_fn(config_box_predictor.conv_hyperparams)
        apply_batch_norm = config_box_predictor.conv_hyperparams.HasField(
            'batch_norm')
        # During training phase, logits are used to compute the loss. Only apply
        # sigmoid at inference to make the inference graph TPU friendly. This is
        # required because during TPU inference, model.postprocess is not called.
        score_converter_fn = build_score_converter(config_box_predictor.score_converter, is_training)
        # Optionally apply clipping to box encodings, when box_encodings_clip_range
        # is set.
        box_encodings_clip_range = None
        if config_box_predictor.HasField('box_encodings_clip_range'):
            box_encodings_clip_range = BoxEncodingsClipRange(
                min=config_box_predictor.box_encodings_clip_range.min,
                max=config_box_predictor.box_encodings_clip_range.max)
        keyword_args = None

        return build_weight_shared_convolutional_keras_box_predictor(
            is_training=is_training,
            num_classes=num_classes,
            conv_hyperparams=conv_hyperparams,
            freeze_batchnorm=freeze_batchnorm,
            inplace_batchnorm_update=inplace_batchnorm_update,
            num_predictions_per_location_list=num_predictions_per_location_list,
            depth=config_box_predictor.depth,
            num_layers_before_predictor=(
                config_box_predictor.num_layers_before_predictor),
            box_code_size=config_box_predictor.box_code_size,
            kernel_size=config_box_predictor.kernel_size,
            add_background_class=add_background_class,
            class_prediction_bias_init=(
                config_box_predictor.class_prediction_bias_init),
            use_dropout=config_box_predictor.use_dropout,
            dropout_keep_prob=config_box_predictor.dropout_keep_probability,
            share_prediction_tower=config_box_predictor.share_prediction_tower,
            apply_batch_norm=apply_batch_norm,
            use_depthwise=config_box_predictor.use_depthwise,
            score_converter_fn=score_converter_fn,
            box_encodings_clip_range=box_encodings_clip_range,
            keyword_args=keyword_args)

    raise ValueError('Unknown box predictor for Keras: {}'.format(box_predictor_oneof))
