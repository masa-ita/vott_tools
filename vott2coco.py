# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import pathlib
import sys
import re
from datetime import datetime
import json
import itertools
import random
import urllib


def load_json(path):
    with open(path, 'r') as f:
        vott_json = json.load(f)

    if vott_json['sourceConnection']:
        vott_path = pathlib.Path(path)
        annotation_dir = vott_path.parent
        asset_ids = vott_json['assets'].keys()
        for asset_id in asset_ids:
            asset_path = annotation_dir/"{}-asset.json".format(asset_id)
            with open(asset_path, 'r') as f:
                asset = json.load(f)
            vott_json['assets'][asset_id]['regions'] = asset['regions']
            vott_json['assets'][asset_id]['version'] = vott_json['version']

    return vott_json

def tags2categories(vott):
    SUPER_CATEGORY = 'objects'

    categories = []
    for idx, tag in enumerate(vott['tags']):
        category = {}
        category['supercategory'] = SUPER_CATEGORY
        category['id'] = idx + 1
        category['name'] = tag['name']
        categories.append(category)
    return categories

def calculate_ratio(ratio_arg):
    re_ratio = re.compile(r'(?P<train>\d+):(?P<val>\d+)(?::(?P<test>\d+))*')

    ratio_match = re_ratio.match(args.ratio)

    if not ratio_match:
        return None

    n_total = int(ratio_match['train']) + int(ratio_match['val']) + \
        int(ratio_match['test'] if ratio_match['test'] else 0)
    ratio = {}
    ratio['train'] = float(ratio_match['train']) / n_total
    ratio['val'] = float(ratio_match['val']) / n_total
    ratio['test'] = float(ratio_match['test'] if ratio_match['test'] else 0) / n_total
    return ratio

def split_assets(asset_ids, ratio):
    asset_ids = set(asset_ids)
    num_assets = len(asset_ids)
    num_val = int(num_assets * ratio['val'])
    num_test = int(num_assets * ratio['test'])
    test_samples = set(random.sample(asset_ids, num_test))
    train_val = asset_ids - test_samples
    val_samples = set(random.sample(train_val, num_val))
    train_samples = train_val - val_samples
    dataset = {'train': train_samples, 'val': val_samples, 'test': test_samples}
    return dataset

def load_imagesets(imagesets_dir):
    imageset_files = pathlib.Path(imagesets_dir).glob('*.txt')
    imagesets = {}
    for imageset_file in imageset_files:
        with open(imageset_file, 'r') as f:
            image_files = [line.strip() for line in f.readlines()]
        imagesets[imageset_file.stem] = image_files
    return imagesets

def imagesets2datasets(file2id, imagesets):
    datasets = {}
    for name, files in imagesets.items():
        datasets[name] = [file2id[file] for file in files]
    return datasets

def print_dataset_size(datasets):
    sizes = ['{} = {}'.format(subset, len(ids)) for subset, ids in datasets.items() ]
    print('Dataset sizes: ' + ', '.join(sizes))

def polygon_area(p):
    n = len(p)
    area = abs(sum(p[i][0]*p[i-1][1] - p[i][1]*p[i-1][0] for i in range(n)))/2.0
    return area

def create_coco(vott_json, asset_ids, output_path):
    now = datetime.now()

    info = {
        "description": None,
        "url": None,
        "version": "1.0",
        "year": now.year,
        "contributor": None,
        "date_created": now.strftime("%Y/%m/%d")
    }

    licenses = [
        {
            "url": None,
            "id": 1,
            "name": "Unknown License"
        },
    ]

    categories = tags2categories(vott_json)
    cat2id = {cat['name']:cat['id'] for cat in categories}

    image_id = 1
    annotation_id = 1
    images = []
    annotations = []

    for asset_id in asset_ids:
        asset = vott_json['assets'][asset_id]

        image = {}
        image['license'] = 1
        image['file_name'] = urllib.parse.unquote(asset['name'])
        image['coco_url'] = None
        image['height'] = asset['size']['height']
        image['width'] = asset['size']['width']
        image['date_captured'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        image['flicker_url'] = None
        image['id'] = image_id
        images.append(image)
        
        regions = asset['regions']

        for region in regions:
            annotation = {}
            points = [(int(p['x']+0.5), int(p['y']+0.5)) for p in region['points']]
            annotation['area'] = polygon_area(points)
            annotation['iscrowd'] = 0
            annotation['image_id'] = image_id
            annotation['bbox'] = [int(region['boundingBox']['left'] + 0.5),
                                  int(region['boundingBox']['top'] + 0.5),
                                  int(region['boundingBox']['width'] + 0.5),
                                  int(region['boundingBox']['height'] + 0.5)]
            annotation['category_id'] =  cat2id[region['tags'][0]]
            annotation['id'] = annotation_id
            annotation_id += 1
            if region['type'] == 'POLYGON':
                annotation['segmentation'] = [list(itertools.chain.from_iterable(points))]
            else:
                pass
            annotations.append(annotation)

        image_id += 1

    coco_annotations = {}
    coco_annotations['info'] = info
    coco_annotations['licenses'] = licenses
    coco_annotations['images'] = images
    coco_annotations['annotations'] = annotations
    coco_annotations['categories'] = categories

    with open(output_path, 'w') as f:
        json.dump(coco_annotations, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create coco formatted annotation file from VoTT's *.vott, *-asset.json files.")
    parser.add_argument('-f', '--vott_file', help="*.vott file path", required=True)
    parser.add_argument('-o', '--output_dir', help="output directory", required=True)
    parser.add_argument('-p', '--output_prefix', help="coco annotation files' prefix", required=True)
    parser.add_argument('-r', '--ratio', default=None, help="dataset size ratio Ex. 80:10:10 default=None")
    parser.add_argument('-i', '--imagesets_dir', default=None, help="imagesets dir")
    parser.add_argument('--overwrite', help='overwrite output files', action='store_true')

    args = parser.parse_args()

    vott_path = pathlib.Path(args.vott_file)
    annotation_dir = vott_path.parent
    output_dir = pathlib.Path(args.output_dir)
    output_prefix = args.output_prefix
    imagesets_dir = args.imagesets_dir
    overwrite = args.overwrite

    if not vott_path.is_file():
        sys.exit('vott file is not found')

    if not output_dir.is_dir():
        sys.exit('--output_dir is not a directory')

    if args.ratio and args.imagesets_dir:
        sys.exit('--ratio and --imagesets_dir can not set simultaniously')

    vott = load_json(vott_path)
    asset_ids = vott['assets'].keys()

    if args.ratio:
        ratio = calculate_ratio(args.ratio)
        if not ratio:
            sys.exit('ratio must follow pattern like 99:99 or 99:99:99')

        datasets = split_assets(asset_ids, ratio)
    elif args.imagesets_dir:
        imagesets = load_imagesets(args.imagesets_dir)
        file2id = {urllib.parse.unquote(asset['name']):id  for id, asset in vott['assets'].items()}
        datasets = imagesets2datasets(file2id, imagesets)
    else:
        datasets = {'': asset_ids}

    print_dataset_size(datasets)

    for subset in datasets.keys():
        output_path = output_dir.joinpath(output_prefix + subset + '.json')
        if output_path.exists() and not overwrite:
            sys.exit('Output file {} exists. Add --overwrite flag to overwrite.'.format(output_path))

    for subset, asset_ids in datasets.items():
        if asset_ids:
            output_path = output_dir.joinpath(output_prefix + subset + '.json')
            create_coco(vott, asset_ids, output_path)
