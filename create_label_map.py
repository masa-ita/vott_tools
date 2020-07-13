import os
import sys
import pathlib
import json
import argparse
from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem
from google.protobuf import text_format


def get_tags(vott_file):
    with open(vott_file) as f:
        vott = json.load(f)
    
    tags = [t['name'] for t in vott['tags']]
    return tags


def convert_classes(classes, start=1):
    msg = StringIntLabelMap()
    for id, name in enumerate(classes, start=start):
        msg.item.append(StringIntLabelMapItem(id=id, name=name))

    text = str(text_format.MessageToBytes(msg, as_utf8=True), 'utf-8')
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create tfrecord datasets from VoTT's *.vott, *-asset.json files.")
    parser.add_argument('-f', '--vott_file', help="*.vott file path", required=True)
    parser.add_argument('-l', '--label_map_file', help="label_map.pbtxt file path", required=True)
    parser.add_argument('--overwrite', help='overwrite output files', action='store_true')

    args = parser.parse_args()

    vott_path = pathlib.Path(args.vott_file)
    label_map_path = pathlib.Path(args.label_map_file)
    overwrite = args.overwrite

    if not vott_path.is_file():
        sys.exit('vott file is not found')

    if label_map_path.exists() and not overwrite:
        sys.exit('Output file {} exists. Add --overwrite flag to overwrite.'.format(label_map_path))

    tags = get_tags(vott_path)
    txt = convert_classes(tags)
    print(txt)
    with open(label_map_path, 'w') as f:
        f.write(txt)
