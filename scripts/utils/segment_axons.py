#!/usr/bin/env python
"""
Script to segment an image using a SAM checkpoint finetuned for axon seg
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

import bids_utils


def get_predictor(model_type, checkpoint, device):
    if device != 'cpu':
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    else:
        sam_model = sam_model_registry[model_type]()
        with open(checkpoint, 'rb') as f:
            state_dict = torch.load(f, map_location=torch.device('cpu'))
        sam_model.load_state_dict(state_dict)
    sam_model.to(device)
    return SamPredictor(sam_model)

def main(args):
    checkpoint = args['checkpoint']
    if 'vit_b' in checkpoint:
        model_type = 'vit_b'
    else:
        model_type = 'vit_l'
    image_path = args['img']
    device = args['device']
    prompt_with_centroids = args['centroid_file'] != None
    is_myelin_model = args['myelin']
    
    predictor = get_predictor(model_type, checkpoint, device)
    image = cv2.imread(image_path)
    predictor.set_image(image)

    if prompt_with_centroids:
        if is_myelin_model:
            prompts = pd.read_csv(args['centroid_file'])
            prompts = torch.tensor(prompts.drop(columns=['x0', 'y0']).values).to(device)
            # get rid of axon IDs
            prompts = prompts[:, 1:]
            transform = ResizeLongestSide(1024)
            prompts_transformed = transform.apply_boxes_torch(prompts, image.shape[-2:])
            mask, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=prompts_transformed,
                multimask_output=False
            )
            mask = torch.sum(mask, dim=0) > 0
            mask = mask.long().numpy()

        else:
            points, labels = bids_utils.load_centroid_prompts([args['centroid_file']], device)
            mask, _, _ = predictor.predict(
                point_coords=points[0].cpu().numpy(),
                point_labels=labels[0],
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

    if not is_myelin_model:
        fname = f'{Path(image_path).parent / Path(image_path).stem}_axonseg.png'
    else:
        fname = f'{Path(image_path).parent / Path(image_path).stem}_myelinseg.png'
    cv2.imwrite(fname, mask[0] * 255)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("checkpoint", help="Model checkpoint")
    parser.add_argument("img", help="Path to the image")

    # Optional argument which requires a parameter (eg. -d test)
    parser.add_argument("-d", "--device", help="Torch device", default='cpu')
    parser.add_argument("-m", "--myelin", help="Segment myelin instead of axon", default=False, action='store_true')
    parser.add_argument("-c", "--centroid_file", help="Path to CSV file containing axon centroids.", default=None)

    args = parser.parse_args()
    main(vars(args))
