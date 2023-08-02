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

def main(args):
    model_type = args['model_type']
    checkpoint = args['checkpoint']
    image_path = args['img']

    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_model.to(args['device'])
    
    # compute image embedding
    predictor = SamPredictor(sam_model)
    predictor.set_image(image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("img", help="Path to the image")
    parser.add_argument("checkpoint", help="Model checkpoint")

    # Optional argument which requires a parameter (eg. -d test)
    parser.add_argument("-m", "--model-type", default="vit_b", help="Model type")
    parser.add_argument("-d", "--device", help="Torch device", default='cpu')

    args = parser.parse_args()
    main(vars(args))