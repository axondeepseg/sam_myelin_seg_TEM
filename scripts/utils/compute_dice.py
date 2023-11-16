from monai.metrics import DiceMetric, MeanIoU
from pathlib import Path
import numpy as np
import cv2
import torch
import argparse

SUFFIX_MAPPING = {
    'SAM': {
        'AXON': '_TEM_axonseg.png',
        'MYELIN': '_TEM_myelinseg.png'
    },
    'NNUNET': {
        'AXON': '_TEM_segnn-axon.png',
        'MYELIN': '_TEM_segnn-myelin.png'
    },
    'ADS': {
        'AXON': '_TEM_seg-axon.png',
        'MYELIN': '_TEM_seg-myelin.png'
    },
    'BIDS': {
        'AXON': '_TEM_seg-axon-manual.png',
        'MYELIN': '_TEM_seg-myelin-manual.png'
    },
}

def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Segment images using nnUNet')
    parser.add_argument('--path-dataset', type=str, required=True, help='Path to original (BIDS) dataset.')
    parser.add_argument('--path-preds', type=str, required=True, help='Path to predictions.')
    parser.add_argument('--path-baseline-preds', required=True, type=str, help='Path to baseline IVADOMED predictions.')
    parser.add_argument('--segtype', type=str, default='AXON', help='Target class; either AXON or MYELIN; defaults to AXON.')
    parser.add_argument('--is-nnunet', default=False, action='store_true', help='if preds come from nnUNetv2 model. Defaults to false.')

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    segtype = args.segtype.upper()
    assert segtype in ['AXON', 'MYELIN'], 'Please provide a valid target class. Either AXON or MYELIN.'

    gt_base = str(Path(args.path_dataset) / 'derivatives' / 'labels' / 'sub-nyuMouse26' / 'micr' / 'sub-nyuMouse26_sample-000')
    fname_base = str(Path(args.path_preds) / 'sub-nyuMouse26_sample-000')
    ads_fname_base = str(Path(args.path_baseline_preds) / 'sub-nyuMouse26_sample-000')
    target_suffix = '_TEM_myelinseg.png' if not args.is_nnunet else '_TEM_segnn.png'
    ads_suffix = '_TEM_seg-myelin.png'
    gt_suffix = '_TEM_seg-myelin-manual.png'
    target_model_type = 'SAM' if not args.is_nnunet else 'NNUNET'
    target_suffix = SUFFIX_MAPPING[target_model_type][segtype]
    ads_suffix = SUFFIX_MAPPING['ADS'][segtype]
    gt_suffix = SUFFIX_MAPPING['BIDS'][segtype]
    
    metric = DiceMetric()
    metric2 = MeanIoU()
    mean_sam, mean_ads = 0, 0
    mean2_sam, mean2_ads = 0, 0
    print('         TARGET            BASELINE (IVADOMED)')
    for sample in range(1, 9):
        sam_fname = fname_base + str(sample) + target_suffix
        ads_fname = ads_fname_base + str(sample) + ads_suffix
        gt_fname = gt_base + str(sample) + gt_suffix

        pred = cv2.imread(sam_fname).transpose(2,0,1) // 255
        ads = cv2.imread(ads_fname).transpose(2,0,1) // 255
        gt = cv2.imread(gt_fname).transpose(2,0,1) // 255

        sam_score = metric(torch.from_numpy(pred)[None], torch.from_numpy(gt)[None])[0][0]
        ads_score = metric(torch.from_numpy(ads)[None], torch.from_numpy(gt)[None])[0][0]
        mean_sam += sam_score
        mean_ads += ads_score

        sam_score2 = metric2(torch.from_numpy(pred)[None], torch.from_numpy(gt)[None])[0][0]
        ads_score2 = metric2(torch.from_numpy(ads)[None], torch.from_numpy(gt)[None])[0][0]
        mean2_sam += sam_score2
        mean2_ads += ads_score2
        print('DICE: ', sam_score.item(), ads_score.item())
        print('IoU:  ', sam_score2.item(), ads_score2.item())
    print(f'TARGET MEAN DICE: {mean_sam / 8}')
    print(f'BASELINE MEAN DICE: {mean_ads / 8}')
    print(f'TARGET MEAN IoU: {mean2_sam / 8}')
    print(f'BASELINE MEAN IoU: {mean2_ads / 8}')    


if __name__ == '__main__':
    main()
