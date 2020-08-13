import os
import sys
import pathlib
import json
import urllib
import re
import random
import skimage.draw
import tensorflow as tf
import numpy as np
import argparse

if tf.__version__ < '2.0.0':
    tf.enable_eager_execution()

def get_categories(vott_file):
    with open(vott_file) as f:
        vott = json.load(f)
    
    categories = []

    for idx, tag in enumerate(vott['tags']):
        category = {}
        category['supercategory'] = 'objects'
        category['id'] = idx + 1
        category['name'] = tag['name']
        categories.append(category)
    return categories

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_tf_example(image_path, asset_json_file, class_id, include_tags, new_size=None):
    with open(str(asset_json_file), 'r') as f:
        example_dict = json.load(f)
    filename = urllib.parse.unquote(example_dict['asset']['name'])
    print(filename)
    height = example_dict['asset']['size']['height']
    width = example_dict['asset']['size']['width']
    image_format = b'jpeg'
    encoded_image_data = tf.io.read_file(str(image_path/filename))
    decoded_jpeg = tf.io.decode_jpeg(encoded_image_data, channels=3)
    if new_size:
        new_height = new_size[0]
        new_width = new_size[1]
        tf_new_size = tf.constant([new_height, new_width], dtype=tf.int32)
        decoded_jpeg = tf.image.resize(decoded_jpeg, tf_new_size, method=tf.image.ResizeMethod.AREA)
    else:
        new_height = height
        new_width = width
    encoded_jpeg = tf.io.encode_jpeg(tf.cast(decoded_jpeg, tf.uint8))
        
    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)
    
    masks = [] # List of encoded PNG images of masks (1 per mask)
    
    regions = example_dict['regions']
    
    for region in regions:
        text = region['tags'][0]
        if text in include_tags:
            xmins.append(region['boundingBox']['left'] / width)
            xmaxs.append((region['boundingBox']['left'] + region['boundingBox']['width'] - 1) / width)
            ymins.append(region['boundingBox']['top'] / height)
            ymaxs.append((region['boundingBox']['top'] + region['boundingBox']['height'] - 1) / height)
            classes_text.append(text.encode('utf-8'))
            classes.append(class_id[text])
            if region['type'] == 'POLYGON':
                mask = np.zeros([new_height, new_width, 1], dtype=np.uint8)
                all_points_x = [int(point['x'] * new_width / width) for point in region['points']]
                all_points_y = [int(point['y'] * new_height / height) for point in region['points']]
                rr, cc = skimage.draw.polygon(all_points_y, all_points_x)
                mask[rr, cc, 0] = 255
                masks.append(tf.image.encode_png(mask).numpy())
    features = {
        'image/height': int64_feature(new_height),
        'image/width': int64_feature(new_width),
        'image/filename': bytes_feature(filename.encode('utf-8')),
        'image/source_id': bytes_feature(filename.encode('utf-8')),
        'image/encoded': bytes_feature(encoded_jpeg.numpy()),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes)
    }
    if masks:
        features['image/object/mask'] = bytes_list_feature(masks)
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    return tf_example
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create tfrecord datasets from VoTT's *.vott, *-asset.json files.")
    parser.add_argument('-f', '--vott_file', help="*.vott file path", required=True)
    parser.add_argument('-i', '--image_dir', help="the directory contains images", required=True)
    parser.add_argument('-o', '--output_dir', help="output directory", required=True)
    parser.add_argument('-p', '--output_prefix', help="tfrecord files' prefix", required=True)
    parser.add_argument('-r', '--ratio', default=None, help="dataset size ratio Ex. 80:10:10 default=None")
    parser.add_argument('-n', '--new_size', default=None, nargs=2, metavar=('height', 'width'), help="new size (height, width)")
    parser.add_argument('-e', '--exclude_tags', default=None, help="exclude tags")
    parser.add_argument('-s', '--select_tags', default=None,  help="select tags")
    parser.add_argument('--overwrite', help='overwrite output files', action='store_true')

    args = parser.parse_args()

    vott_path = pathlib.Path(args.vott_file)
    annotation_dir = vott_path.parent
    image_dir = pathlib.Path(args.image_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_prefix = args.output_prefix
    new_size = [int(n) for n in args.new_size] if args.new_size else None
    if args.exclude_tags:
        exclude_tags = set([tag.strip for tag in args.exclude_tags.split(',')])
        print("exclude:{}".format(exclude_tags))
    else:
        exclude_tags = set()

    if args.select_tags:
        select_tags = set([tag.strip for tag in args.select_tags.split(',')])
        print("exclude:{}".format(select_tags))
    else:
        select_tags = set()

    overwrite = args.overwrite

    if not vott_path.is_file():
        sys.exit('vott file is not found')

    if not image_dir.is_dir():
        sys.exit('--image_dir is not a directory')

    if not output_dir.is_dir():
        sys.exit('--output_dir is not a directory')

    asset_files = set(annotation_dir.glob('*-asset.json'))
    categories = get_categories(vott_path)
    cat2id = {cat['name']:cat['id'] for cat in categories}

    if select_tags:
        include_tags = select_tags
    else:
        include_tags = set([cat['name'] for cat in categories])
    include_tags = include_tags - exclude_tags 
    print("output tags {} only".format(include_tags))

    if args.ratio:
        re_ratio = re.compile(r'(?P<train>\d+):(?P<val>\d+)(?::(?P<test>\d+))*')

        ratio_match = re_ratio.match(args.ratio)

        if not ratio_match:
            sys.exit('ratio must follow pattern like 99:99 or 99:99:99')

        n_total = int(ratio_match['train']) + int(ratio_match['val']) + \
            int(ratio_match['test'] if ratio_match['test'] else 0)
        ratio = {}
        ratio['train'] = float(ratio_match['train']) / n_total
        ratio['val'] = float(ratio_match['val']) / n_total
        ratio['test'] = float(ratio_match['test'] if ratio_match['test'] else 0) / n_total

        for suffix in ['train', 'val', 'test']:
            output_path = output_dir.joinpath(output_prefix + suffix + '.tfrecord')
            if output_path.exists() and not overwrite:
                sys.exit('Output file {} exists. Add --overwrite flag to overwrite.'.format(output_path))

        num_assets = len(asset_files)
        num_val = int(num_assets * ratio['val'])
        num_test = int(num_assets * ratio['test'])
        test_samples = set(random.sample(asset_files, num_test))
        train_val = asset_files - test_samples
        val_samples = set(random.sample(train_val, num_val))
        train_samples = train_val - val_samples
        num_train = len(train_samples)

        dataset = {'train': train_samples, 'val': val_samples, 'test': test_samples}

        print('Num Samples: {} (train -> {}, validation -> {}, test -> {})'.format(num_assets, num_train, num_val, num_test))

        for subset, samples in dataset.items():
            if samples:
                output_path = output_dir.joinpath(output_prefix + subset + '.tfrecord')
                writer = tf.io.TFRecordWriter(str(output_path))

                for sample in samples:
                    tf_example = create_tf_example(image_dir, sample, cat2id, include_tags, new_size)
                    writer.write(tf_example.SerializeToString())

                writer.close()

    else:
        output_path = output_dir.joinpath(output_prefix + '.tfrecord')
        if output_path.exists() and not overwrite:
            sys.exit('Output file {} exists. Add --overwrite flag to overwrite.'.format(output_path))

        writer = tf.io.TFRecordWriter(str(output_path))

        for sample in asset_files:
            tf_example = create_tf_example(image_dir, sample, cat2id, include_tags, new_size)
            writer.write(tf_example.SerializeToString())

        writer.close()
