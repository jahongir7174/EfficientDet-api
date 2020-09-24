from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
from os.path import join

import PIL.Image
import tensorflow.compat.v1 as tf
from lxml import etree

from nets.utils import dataset_util, label_map_util
from utils import config

SETS = ['train', 'val', 'trainval', 'test']


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False):
    img_path = join(config.image_dir, data['filename'])
    full_path = join(dataset_directory, img_path + '.jpg')
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    x_min = []
    y_min = []
    x_max = []
    y_max = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    if 'object' in data:
        for obj in data['object']:
            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue

            difficult_obj.append(int(difficult))

            x_min.append(float(obj['bndbox']['xmin']) / width)
            y_min.append(float(obj['bndbox']['ymin']) / height)
            x_max.append(float(obj['bndbox']['xmax']) / width)
            y_max.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map_dict[obj['name']])
            truncated.append(int(obj['truncated']))
            poses.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(x_min),
        'image/object/bbox/xmax': dataset_util.float_list_feature(x_max),
        'image/object/bbox/ymin': dataset_util.float_list_feature(y_min),
        'image/object/bbox/ymax': dataset_util.float_list_feature(y_max),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated)}))
    return example


def main():
    for _set in ['train', 'val']:
        if _set not in SETS:
            raise ValueError('set must be in : {}'.format(SETS))

        writer = tf.python_io.TFRecordWriter(join(config.data_dir, _set + '.tf'))

        label_map_dict = label_map_util.get_label_map_dict(config.label_map_path)

        print('Reading from Dubai dataset.')
        examples_path = join(config.data_dir, _set + '.txt')
        annotations_dir = join(config.data_dir, config.label_dir)
        examples_list = dataset_util.read_examples_list(examples_path)
        for idx, example in enumerate(examples_list):
            if idx % 100 == 0:
                print('On image %d of %d', idx, len(examples_list))
            path = join(annotations_dir, example + '.xml')
            with tf.gfile.GFile(path, 'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

            tf_example = dict_to_tf_example(data,
                                            config.data_dir,
                                            label_map_dict)
            writer.write(tf_example.SerializeToString())

        writer.close()


if __name__ == '__main__':
    main()
