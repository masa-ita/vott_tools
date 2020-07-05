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

parser = argparse.ArgumentParser(description="Create coco formatted annotation file from VoTT's *.vott, *-asset.json files.")
parser.add_argument('-f', '--vott_file', help="*.vott file path", required=True)
parser.add_argument('-o', '--output_dir', help="output directory", required=True)
parser.add_argument('-p', '--output_prefix', help="coco annotation files' prefix", required=True)
parser.add_argument('-r', '--ratio', default=None, help="dataset size ratio Ex. 80:10:10 default=None")
parser.add_argument('--overwrite', help='overwrite output files', action='store_true')

args = parser.parse_args()

vott_path = pathlib.Path(args.vott_file)
annotation_dir = vott_path.parent
output_dir = pathlib.Path(args.output_dir)
output_prefix = args.output_prefix
overwrite = args.overwrite

if not vott_path.is_file():
    sys.exit('vott file is not found')

if not output_dir.is_dir():
    sys.exit('--output_dir is not a directory')

re_ratio = re.Regex(r'(?P<train>\d+):(?P<val>\d+):(?P<test>\d)')

for suffix in ['train', 'val', 'test']:
    output_path = output_dir.joinpath(output_prefix + suffix + '.json')
    if output_path.exists() and not overwrite:
        sys.exit('Output file {} exists. Add --overwrite flag to overwrite.'.format(output_path))

SUPER_CATEGORY = 'objects'

asset_files = set(annotation_dir.glob('*-asset.json'))
num_assets = len(asset_files)
num_val = int(num_assets * val_ratio)
num_test = int(num_assets * test_ratio)
test_samples = set(random.sample(asset_files, num_test))
train_val = asset_files - test_samples
val_samples = set(random.sample(train_val, num_val))
train_samples = train_val - val_samples
num_train = len(train_samples)

dataset = {'train': train_samples, 'val': val_samples, 'test': test_samples}

print('Num Samples: {} (train -> {}, validation -> {}, test -> {})'.format(num_assets, num_train, num_val, num_test))

now = datetime.now()

info = {
    "description": "Dataset",
    "url": "http://",
    "version": "1.0",
    "year": now.year,
    "contributor": "",
    "date_created": now.strftime("%Y/%m/%d")
}

licenses = [
    {
        "url": "",
        "id": 1,
        "name": "Unknown License"
    },
]

def get_categories(vott_file):
    with open(vott_file) as f:
        vott = json.load(f)
    
    categories = []

    for idx, tag in enumerate(vott['tags']):
        category = {}
        category['supercategory'] = SUPER_CATEGORY
        category['id'] = idx + 1
        category['name'] = tag['name']
        categories.append(category)
    return categories

def polygon_area(p):
    n = len(p)
    area = abs(sum(p[i][0]*p[i-1][1] - p[i][1]*p[i-1][0] for i in range(n)))/2.0
    return area

categories = get_categories(vott_path)

cat2id = {cat['name']:cat['id'] for cat in categories}

def create_coco(output_path, samples):
    image_id = 1
    annotation_id = 1
    images = []
    annotations = []

    for json_path in samples:
        with open(json_path) as f:
            asset = json.load(f)

        image = {}
        image['license'] = 1
        image['file_name'] = asset['asset']['name']
        image['coco_url'] = None
        image['height'] = asset['asset']['size']['height']
        image['width'] = asset['asset']['size']['width']
        image['date_captured'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        image['flicker_url'] = None
        image['id'] = image_id
        images.append(image)
        
        regions = asset['regions']

        for region in regions:
            if region['type'] == 'POLYGON':
                annotation = {}
                points = [(int(p['x']+0.5), int(p['y']+0.5)) for p in region['points']]
                annotation['segmentation'] = [list(itertools.chain.from_iterable(points))]
                annotation['area'] = polygon_area(points)
                annotation['iscrowd'] = 0
                annotation['image_id'] = image_id
                annotation['bbox'] = [region['boundingBox']['left'],
                                    region['boundingBox']['top'],
                                    region['boundingBox']['width'],
                                    region['boundingBox']['height']]
                annotation['category_id'] =  cat2id[region['tags'][0]]
                annotation['id'] = annotation_id
                annotation_id += 1
            elif region['type'] == 'RECTANGLE':
                annotation = {}
                points = [(int(p['x']+0.5), int(p['y']+0.5)) for p in region['points']]
                annotation['area'] = polygon_area(points)
                annotation['iscrowd'] = 0
                annotation['image_id'] = image_id
                annotation['bbox'] = [region['boundingBox']['left'],
                                    region['boundingBox']['top'],
                                    region['boundingBox']['width'],
                                    region['boundingBox']['height']]
                annotation['category_id'] =  cat2id[region['tags'][0]]
                annotation['id'] = annotation_id
                annotation_id += 1
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

for subset, samples in dataset.items():
    if samples:
        output_path = output_dir.joinpath(output_prefix + subset + '.json')

        create_coco(output_path, samples)
