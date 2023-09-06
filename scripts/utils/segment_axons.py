#!/usr/bin/env python
"""
Script to segment an image using a SAM checkpoint finetuned for axon seg
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
from segment_anything import SamPredictor, sam_model_registry

import bids_utils


def get_predictor(model_type, checkpoint, device):
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_model.to(device)
    return SamPredictor(sam_model)

def main(args):
    model_type = args['model_type']
    checkpoint = args['checkpoint']
    image_path = args['img']
    device = args['device']
    prompt_with_centroids = args['centroid_file'] != None
    
    predictor = get_predictor(model_type, checkpoint, device)
    image = cv2.imread(image_path)
    predictor.set_image(image)

    if prompt_with_centroids:
        points, labels = bids_utils.load_centroid_prompts([args['centroid_file']], device)
        print("POINTS SHAPE -> ", points.shape)
        mask, _, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=None,
            multimask_output=False
        )
    else:
        # get full bbox prompt
        H, W = predictor.original_size
        prompt = np.array([[0,0,W-1,H-1]])
        mask, _, _ = predictor.predict(
            point_coords=None,
            box=prompt,
            multimask_output=False
        )

    fname = f'{Path(image_path).parent / Path(image_path).stem}_axonseg.png'
    cv2.imwrite(fname, mask[0] * 255)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("img", help="Path to the image")
    parser.add_argument("checkpoint", help="Model checkpoint")

    # Optional argument which requires a parameter (eg. -d test)
    parser.add_argument("-m", "--model_type", default="vit_b", help="Model type")
    parser.add_argument("-d", "--device", help="Torch device", default='cpu')
    parser.add_argument("-c", "--centroid_file", help="Path to CSV file containing axon centroids.", default=None)

    args = parser.parse_args()
    main(vars(args))
