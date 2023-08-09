#!/usr/bin/env python3
"""
This script splits the TEM images in 2 regions. The original size is 
2286x3762 but we split them to use more efficiently the 1024x1024 input 
size of the transformer.
"""

__author__ = "Armand Collin"
__license__ = "MIT"

import cv2
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

from bids_utils import index_bids_dataset


IVADOMED_VALIDATION_SUBJECTS = ['sub-nyuMouse10']
IVADOMED_TEST_SUBJECTS = ['sub-nyuMouse26']

def main(datapath, out_path):
    datapath = Path(datapath)
    data_dict = index_bids_dataset(datapath)
    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True)

    for sub in tqdm(data_dict.keys()):
        px_size = data_dict[sub]['px_size']
        samples = (s for s in data_dict[sub].keys() if 'sample' in s)
        for sample in samples:
            # load image and gt
            img = cv2.imread(data_dict[sub][sample]['image'])
            gt = cv2.imread(data_dict[sub][sample]['axon'])
            # split in 2
            width = img.shape[1]
            img_a = img[:,:width//2]
            img_b = img[:, width//2:]
            gt_a = gt[:,:width//2]
            gt_b = gt[:, width//2:]
            
            # chose output folder
            if sub in IVADOMED_VALIDATION_SUBJECTS:
                output_target = out_path / 'val'
            elif sub in IVADOMED_TEST_SUBJECTS:
                output_target = out_path / 'test'
            else:
                output_target = out_path / 'train' 
            output_target.mkdir(exist_ok=True)

            split_path = output_target / 'imgs'
            split_path.mkdir(exist_ok=True)
            gt_path = output_target / 'gts'
            gt_path.mkdir(exist_ok=True)
            
            # save both halves in output folder
            img_fname = str(split_path / f'{sub}_{sample}')
            cv2.imwrite(f'{img_fname}a_TEM.png', img_a)
            cv2.imwrite(f'{img_fname}b_TEM.png', img_b)
            
            gt_fname = str(gt_path / f'{sub}_{sample}')
            cv2.imwrite(f'{gt_fname}a_TEM_Seg-axon-manual.png', gt_a)
            cv2.imwrite(f'{gt_fname}b_TEM_Seg-axon-manual.png', gt_b)
    print('Preprocessing done.')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '-d',
        dest='datapath',
        type=str,
        required=True,
        help='Path to the BIDS dataset'
    )
    parser.add_argument(
        '-o',
        dest='output_path',
        type=str,
        required=True,
        help='Where to save the output'
    )

    args = parser.parse_args()
    main(args.datapath, args.output_path)