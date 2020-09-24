import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from os.path import join, exists
from utils import config, util

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from nets import model_lib_v2


def main():
    pipeline_path = join('weights', f'D{config.phi}', f'd{config.phi}.config')
    tf.config.set_soft_device_placement(True)
    strategy = tf.compat.v2.distribute.MirroredStrategy()
    nb_gpu = strategy.num_replicas_in_sync
    if not exists(pipeline_path):
        util.download_extract(nb_gpu)
    with strategy.scope():
        model_lib_v2.train_loop(pipeline_path,
                                model_dir=join('weights', f'D{config.phi}'))


if __name__ == '__main__':
    main()
