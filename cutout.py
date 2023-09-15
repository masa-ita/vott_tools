import sys
import pathlib
import json
import urllib
import cv2
import numpy as np
import argparse


def create_masked_image(image_path, asset_json_file, output_path, new_size=None):
    with open(str(asset_json_file), 'r') as f:
        example_dict = json.load(f)
    filename = urllib.parse.unquote(example_dict['asset']['name'])
    print(filename)
    height = example_dict['asset']['size']['height']
    width = example_dict['asset']['size']['width']
    decoded_jpeg = cv2.imread(str(image_path.joinpath(filename)))
    if new_size:
        new_height = new_size[0]
        new_width = new_size[1]
        decoded_jpeg = cv2.resize(decoded_jpeg, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    else:
        new_height = height
        new_width = width
            
    regions = example_dict['regions']
    print(f"Found {len(regions)} regions.")

    mask = np.zeros_like(decoded_jpeg)

    for region in regions:
        if region['type'] == 'POLYGON':
            all_points = np.array([[point['x'], point['y']] for point in region['points']], np.int32)
            cv2.fillPoly(mask, [all_points], (255,255,255))
    result = cv2.bitwise_and(decoded_jpeg, mask)
    cv2.imwrite(str(output_path.joinpath(filename)), result, [cv2.IMWRITE_JPEG_QUALITY, 75])
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create masked image files from VoTT's *.vott, *-asset.json files.")
    parser.add_argument('-f', '--vott_file', help="*.vott file path", required=True)
    parser.add_argument('-i', '--image_dir', help="the directory contains images", required=True)
    parser.add_argument('-o', '--output_dir', help="output directory", required=True)
    parser.add_argument('-n', '--new_size', default=None, nargs=2, metavar=('height', 'width'), help="new size (height, width)")
    parser.add_argument('--overwrite', help='overwrite output files', action='store_true')

    args = parser.parse_args()

    vott_path = pathlib.Path(args.vott_file)
    annotation_dir = vott_path.parent
    image_dir = pathlib.Path(args.image_dir)
    output_dir = pathlib.Path(args.output_dir)
    new_size = [int(n) for n in args.new_size] if args.new_size else None
    overwrite = args.overwrite
    print(new_size)

    if not vott_path.is_file():
        sys.exit('vott file is not found')

    if not image_dir.is_dir():
        sys.exit('--image_dir is not a directory')

    if not output_dir.is_dir():
        sys.exit('--output_dir is not a directory')

    asset_files = set(annotation_dir.glob('*-asset.json'))

    for sample in asset_files:
        create_masked_image(image_dir, sample, output_dir, new_size)
