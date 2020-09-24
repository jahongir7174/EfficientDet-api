from nets.core import losses
from nets.protos import losses_pb2


def build(loss_config):
    classification_loss = _build_classification_loss(loss_config.classification_loss)
    localization_loss = _build_localization_loss(loss_config.localization_loss)
    classification_weight = loss_config.classification_weight
    localization_weight = loss_config.localization_weight

    return (classification_loss,
            localization_loss,
            classification_weight,
            localization_weight,
            None,
            None,
            None)


def _build_localization_loss(loss_config):
    """Builds a localization loss based on the loss config.

    Args:
      loss_config: A losses_pb2.LocalizationLoss object.

    Returns:
      Loss based on the config.

    Raises:
      ValueError: On invalid loss_config.
    """
    if not isinstance(loss_config, losses_pb2.LocalizationLoss):
        raise ValueError('loss_config not of type losses_pb2.LocalizationLoss.')

    loss_type = loss_config.WhichOneof('localization_loss')

    if loss_type == 'weighted_l2':
        return losses.WeightedL2LocalizationLoss()

    if loss_type == 'weighted_smooth_l1':
        return losses.WeightedSmoothL1LocalizationLoss(
            loss_config.weighted_smooth_l1.delta)

    if loss_type == 'weighted_iou':
        return losses.WeightedIOULocalizationLoss()

    if loss_type == 'l1_localization_loss':
        return losses.L1LocalizationLoss()

    raise ValueError('Empty loss config.')


def _build_classification_loss(loss_config):
    """Builds a classification loss based on the loss config.

    Args:
      loss_config: A losses_pb2.ClassificationLoss object.

    Returns:
      Loss based on the config.

    Raises:
      ValueError: On invalid loss_config.
    """
    if not isinstance(loss_config, losses_pb2.ClassificationLoss):
        raise ValueError('loss_config not of type losses_pb2.ClassificationLoss.')

    loss_type = loss_config.WhichOneof('classification_loss')

    if loss_type == 'weighted_sigmoid':
        return losses.WeightedSigmoidClassificationLoss()

    if loss_type == 'weighted_sigmoid_focal':
        config = loss_config.weighted_sigmoid_focal
        alpha = None
        if config.HasField('alpha'):
            alpha = config.alpha
        return losses.SigmoidFocalClassificationLoss(
            gamma=config.gamma,
            alpha=alpha)

    if loss_type == 'weighted_softmax':
        config = loss_config.weighted_softmax
        return losses.WeightedSoftmaxClassificationLoss(
            logit_scale=config.logit_scale)

    if loss_type == 'weighted_logits_softmax':
        config = loss_config.weighted_logits_softmax
        return losses.WeightedSoftmaxClassificationAgainstLogitsLoss(
            logit_scale=config.logit_scale)

    if loss_type == 'bootstrapped_sigmoid':
        config = loss_config.bootstrapped_sigmoid
        return losses.BootstrappedSigmoidClassificationLoss(
            alpha=config.alpha,
            bootstrap_type=('hard' if config.hard_bootstrap else 'soft'))

    if loss_type == 'penalty_reduced_logistic_focal_loss':
        config = loss_config.penalty_reduced_logistic_focal_loss
        return losses.PenaltyReducedLogisticFocalLoss(
            alpha=config.alpha, beta=config.beta)

    raise ValueError('Empty loss config.')
