import tensorflow as tf
from object_detection.utils import dataset_util
import xml.etree.ElementTree
import numpy as np
import cv2
import os

OUTPUT_PATH = '/home/bruno/udacity/carnd3/RealSDC/Test_Images/Simulator/test/samples.tfrec'
SAMPLE_DIR = '/home/bruno/udacity/carnd3/RealSDC/Test_Images/Simulator/test'

LABELS_DICTIONARY = {"trafficlight": 10, }


def create_tf_example(example, image):

    imagefullpath = os.path.join(SAMPLE_DIR, image)
    imagefullpathencoded = imagefullpath.encode()
    xmlfullpath = os.path.join(SAMPLE_DIR, example)
    img = cv2.imread(imagefullpath, 0)
    height, width = img.shape[:2]

    with tf.gfile.GFile(imagefullpath, 'rb') as fid:
        encoded_image = fid.read()

    image_format = None
    if imagefullpath.endswith('jpg'):
        image_format = 'jpg'.encode()
    if imagefullpath.endswith('png'):
        image_format = 'png'.encode()

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    e = xml.etree.ElementTree.parse(xmlfullpath).getroot()

    for obj in e.findall('object'):
        box = obj.find('bndbox')
        xmin = float(box.find('xmin').text) / width
        xmax = float(box.find('xmax').text) / width
        ymin = float(box.find('ymin').text) / height
        ymax = float(box.find('ymax').text) / height
        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)

        class_text = (obj.find('name').text).encode()
        classes_text.append(class_text)

        classes.append(int(LABELS_DICTIONARY['trafficlight']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(imagefullpathencoded),
        'image/source_id': dataset_util.bytes_feature(imagefullpathencoded),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(OUTPUT_PATH)

    xmlfiles = []
    images = []
    for file in os.listdir(SAMPLE_DIR):
        if file.endswith('.xml'):
            xmlfiles.append(file)
        if file.endswith('.jpg') or file.endswith('.png'):
            images.append(file)

    for example, image in zip(xmlfiles, images):
        tf_example = create_tf_example(example, image)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
