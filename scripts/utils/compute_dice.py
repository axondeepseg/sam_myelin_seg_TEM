from monai.metrics import DiceMetric
import numpy as np
import cv2
import torch

def main():

    gt_dir = '/home/herman/Documents/NEUROPOLY_21/datasets/data_axondeepseg_tem/derivatives/labels/sub-nyuMouse26/micr/sub-nyuMouse26_sample-000'
    fname_base = 'test_images/sub-nyuMouse26_sample-000'
    sam_suffix = '_TEM_axonseg.png'
    ads_suffix = '_TEM_seg-axon.png'
    gt_suffix = '_TEM_seg-axon-manual.png'
    
    metric = DiceMetric()
    mean_sam, mean_ads = 0, 0
    for sample in range(1, 9):
        sam_fname = fname_base + str(sample) + sam_suffix
        ads_fname = fname_base + str(sample) + ads_suffix
        gt_fname = gt_dir + str(sample) + gt_suffix

        pred = cv2.imread(sam_fname).transpose(2,0,1) // 255
        ads = cv2.imread(ads_fname).transpose(2,0,1) // 255
        gt = cv2.imread(gt_fname).transpose(2,0,1) // 255

        sam_score = metric(torch.from_numpy(pred)[None], torch.from_numpy(gt)[None])[0][0]
        ads_score = metric(torch.from_numpy(ads)[None], torch.from_numpy(gt)[None])[0][0]
        mean_sam += sam_score
        mean_ads += ads_score
        print(sam_score, ads_score)
    print(f'SAM MEAN: {mean_sam / 8}')
    print(f'ADS MEAN: {mean_ads / 8}')  

if __name__ == '__main__':
    main()