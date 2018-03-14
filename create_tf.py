#!/usr/bin/env python3
import glob
import sys
import math
import os
import io
import xml.etree.ElementTree as ET

import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict


# % of eval records
RATIO = 0.8
IMG_SRC = "tmp/images_output/JPEGImages"
XML_SRC = "tmp/images_output/Annotations"


def findfiles(directory, pattern):
    objects = glob.glob(os.path.join(directory, pattern))

    files = []
    for i in objects:
        if isFile(directory + i):
            files.append(i)
    return files

def isFile(object):
    try:
        os.listdir(object)
        return False
    except Exception:
        return True


def getLabels(path):
    files = findfiles(path, "*.xml")
    lst = []
    for f in files:
        fin = open(f)
        try:
            tree = ET.fromstring(fin.read())
            objs = tree.findall('object')
            for obj in objs:
                lst.append({
                    "filename": tree.find("filename").text,
                    "width": int(tree.find('size').find("width").text),
                    "height": int(tree.find('size').find("height").text),
                    "class":  obj.find("name").text,
                    "xmin": int(obj.find("bndbox").find("xmin").text),
                    "ymin": int(obj.find("bndbox").find("ymin").text),
                    "xmax": int(obj.find("bndbox").find("xmax").text),
                    "ymax": int(obj.find("bndbox").find("xmax").text)
                })
        except Exception as e:
            print(e)
        fin.close()
    return lst

# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'apple':
        return 1
    else:
        None

def create_tf_example(obj, path):

    with tf.gfile.GFile(os.path.join(path, '{}.jpg'.format(obj["filename"])), 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    image_format = b'jpg'
    filename = obj["filename"].encode('utf8')
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    xmins.append(obj["xmin"] / width)
    xmaxs.append(obj["xmax"] / width)
    ymins.append(obj["ymin"] / height)
    ymaxs.append(obj["ymax"] / height)
    classes_text.append(obj['class'].encode('utf8'))
    classes.append(class_text_to_int(obj['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main():

    tf_record_out = os.path.join(os.getcwd(), 'data/train')
    tf_eval_out = os.path.join(os.getcwd(), 'data/eval')

    img_path = os.path.join(os.getcwd(), IMG_SRC)
    xml_path = os.path.join(os.getcwd(), XML_SRC)

    lst = getLabels(xml_path)

    r_writer = tf.python_io.TFRecordWriter(tf_record_out)
    e_writer = tf.python_io.TFRecordWriter(tf_eval_out)

    record_len = math.floor(len(lst) * RATIO)

    pos = 0
    for obj in lst:
        if pos < record_len:
            tf_example = create_tf_example(obj, img_path)
            r_writer.write(tf_example.SerializeToString())
        else:
            tf_example = create_tf_example(obj, img_path)
            e_writer.write(tf_example.SerializeToString())
        pos += 1

    r_writer.close()
    e_writer.close()

    print("Created TF records")


main()
