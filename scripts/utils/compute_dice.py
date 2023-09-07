from monai.metrics import DiceMetric, MeanIoU
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
    metric2 = MeanIoU()
    mean_sam, mean_ads = 0, 0
    mean2_sam, mean2_ads = 0, 0
    print('         SAM               IVADOMED')
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

        sam_score2 = metric2(torch.from_numpy(pred)[None], torch.from_numpy(gt)[None])[0][0]
        ads_score2 = metric2(torch.from_numpy(ads)[None], torch.from_numpy(gt)[None])[0][0]
        mean2_sam += sam_score2
        mean2_ads += ads_score2
        print('DICE: ', sam_score.item(), ads_score.item())
        print('IoU:  ', sam_score2.item(), ads_score2.item())
    print(f'SAM MEAN DICE: {mean_sam / 8}')
    print(f'ADS MEAN DICE: {mean_ads / 8}')
    print(f'SAM MEAN IoU: {mean2_sam / 8}')
    print(f'ADS MEAN IoU: {mean2_ads / 8}')    


if __name__ == '__main__':
    main()