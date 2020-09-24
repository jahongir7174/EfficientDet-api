import os
import pickle
from os.path import join, basename
from xml.etree.ElementTree import parse as parse_fn

import cv2
import numpy as np
import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from nets.utils import config_util
from nets.builders import model_builder
from utils.util import parse_annotation
from utils.util import draw_boxes


def detection_function(model):
    @tf.function
    def fn(image):
        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes, )
        detections = model.postprocess(prediction_dict, shapes, )

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return fn


def main():
    threshold = 0.22
    pipeline_config = join('weights', 'D4', 'd4.config')

    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(join('weights', 'D4', 'ckpt-389')).expect_partial()
    detect_fn = detection_function(detection_model)

    f_names = []
    with open(join('..', 'Dataset', 'Dubai', 'val.txt')) as reader:
        lines = reader.readlines()
    for line in lines:
        f_names.append(line.rstrip().split(' ')[0])
    result_dict = {}
    for f_name in tqdm.tqdm(f_names):
        image_path = join('..', 'Dataset', 'Dubai', 'IMAGES', f_name + '.jpg')
        label_path = join('..', 'Dataset', 'Dubai', 'LABELS', f_name + '.xml')
        image = cv2.imread(image_path)
        image = image[:, :, ::-1]

        input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = detect_fn(input_tensor)
        image_np = image.copy()
        scores = detections['detection_scores'][0].numpy()
        pred_boxes = detections['detection_boxes'][0].numpy()
        im_height, im_width, _ = image.shape
        pred_boxes_np = []
        for pred_box in pred_boxes:
            y_min, x_min, y_max, x_max = pred_box
            x_min = int(x_min * im_width)
            y_min = int(y_min * im_height)
            x_max = int(x_max * im_width)
            y_max = int(y_max * im_height)
            pred_boxes_np.append([x_min, y_min, x_max, y_max])
        true_boxes = []
        for element in parse_fn(label_path).getroot().iter('object'):
            true_boxes.append(parse_annotation(element))
        result = {'detection_boxes': pred_boxes_np,
                  'groundtruth_boxes': true_boxes,
                  'confidence': scores}
        result_dict[f'{f_name}.jpg'] = result
        draw_boxes(image_np, pred_boxes, scores, threshold)
        cv2.imwrite(join('results', basename(image_path)), image_np[:, :, ::-1])
    with open(join('results', 'd4.pickle'), 'wb') as writer:
        pickle.dump(result_dict, writer)


if __name__ == '__main__':
    main()
