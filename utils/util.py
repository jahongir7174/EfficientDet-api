import os
import shutil
import tarfile
from os.path import isdir, exists, join

import cv2
from tensorflow.keras.utils import get_file

from utils import config


def draw_boxes(image, boxes, scores, threshold):
    im_height, im_width, _ = image.shape
    for box, score in zip(boxes, scores):
        if score > threshold:
            y_min, x_min, y_max, x_max = box
            x_min = int(x_min * im_width)
            x_max = int(x_max * im_width)
            y_min = int(y_min * im_height)
            y_max = int(y_max * im_height)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)


def find_node(parent, name, debug_name=None, parse=None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            print('illegal value for \'{}\': {}'.format(debug_name, e))
    return result


def parse_annotation(element):
    bbox = find_node(element, 'bndbox')
    x_min = find_node(bbox, 'xmin', 'bndbox.xmin', parse=float) - 1
    y_min = find_node(bbox, 'ymin', 'bndbox.ymin', parse=float) - 1
    x_max = find_node(bbox, 'xmax', 'bndbox.xmax', parse=float) - 1
    y_max = find_node(bbox, 'ymax', 'bndbox.ymax', parse=float) - 1
    return [x_min, y_min, x_max, y_max]


def get_num_classes(label_name):
    from nets.utils import label_map_util
    label_map = label_map_util.load_labelmap(label_name)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())


def download_extract(nb_gpu):
    chosen_model = 'efficientdet-d0'

    model_name = f'efficientdet_d{config.phi}_coco17_tpu-32'
    pretrained_checkpoint = f'efficientdet_d{config.phi}_coco17_tpu-32.tar.gz'
    base_pipeline_file = 'pipeline.config'
    batch_size = config.batch_size
    if exists(join('weights', f'D{config.phi}')) and isdir(join('weights', f'D{config.phi}')):
        shutil.rmtree(join('weights', f'D{config.phi}'))
    os.makedirs(join('weights', f'D{config.phi}'))
    tar_path = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/' + pretrained_checkpoint
    downloaded_path = get_file(join(os.getcwd(), "weights", f'D{config.phi}', chosen_model),
                               tar_path,
                               untar=True)
    tar_path = downloaded_path + '.tar.gz'
    tar = tarfile.open(tar_path)
    tar.extractall(join("weights", f'D{config.phi}'))
    tar.close()
    if exists(tar_path):
        os.remove(tar_path)

    pipeline_f_name = join("weights", f'D{config.phi}', model_name, base_pipeline_file)
    fine_tune_checkpoint = f"weights/D{config.phi}/{model_name}/checkpoint/ckpt-0"
    num_classes = get_num_classes(config.label_map_path)
    train_record_f_name = config.data_dir + '/train.tf'
    test_record_f_name = config.data_dir + '/val.tf'
    with open(pipeline_f_name) as reader:
        s = reader.read()
    learning_rate_base = 0.08 * nb_gpu * config.batch_size / 128.
    warmup_learning_rate = 0.0010000000474974513
    if learning_rate_base < warmup_learning_rate:
        learning_rate_base = warmup_learning_rate
    with open(join('weights', f'D{config.phi}', f'd{config.phi}.config'), 'w') as writer:
        s = s.replace('fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED"',
                      f'fine_tune_checkpoint: "{fine_tune_checkpoint}"')

        s = s.replace('input_path: "PATH_TO_BE_CONFIGURED/train2017-?????-of-00256.tfrecord"',
                      f'input_path: "{train_record_f_name}"')

        s = s.replace('input_path: "PATH_TO_BE_CONFIGURED/val2017-?????-of-00032.tfrecord"',
                      f'input_path: "{test_record_f_name}"')

        s = s.replace('label_map_path: "PATH_TO_BE_CONFIGURED/label_map.txt"',
                      f'label_map_path: "{config.label_map_path}"')

        s = s.replace('batch_size: 128', f'batch_size: {batch_size}')

        s = s.replace('num_classes: 90', f'num_classes: {num_classes}')

        s = s.replace('fine_tune_checkpoint_type: "classification"',
                      'fine_tune_checkpoint_type: "detection"')

        s = s.replace('learning_rate_base: 0.07999999821186066',
                      f'learning_rate_base: {learning_rate_base}')
        writer.write(s)
