from pathlib import Path
import json
import pandas as pd
import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from segment_anything.utils.transforms import ResizeLongestSide


def get_sample_bboxes(subject, sample, maps_path):
    prompts_fname = maps_path / subject / 'micr' / f'{subject}_{sample}_prompts.csv'
    prompts_df = pd.read_csv(prompts_fname)
    return prompts_df[['bbox_min_x', 'bbox_min_y', 'bbox_max_x', 'bbox_max_y']]

def get_myelin_bbox(bbox_df, axon_id):
    return np.array(bbox_df.iloc[axon_id])

def get_myelin_map(subject, sample, maps_path):
    map_fname = maps_path / subject / 'micr' / f'{subject}_{sample}_myelinmap.png'
    return cv2.imread(str(map_fname))
    
def get_myelin_mask(myelin_map, axon_id):
    return 255 * (myelin_map == axon_id + 1)

def load_centroid_prompts(csv_paths, device):
    '''
    Loads axon centroids from CSV derivative file to prompt SAM,
    which expects a tuple with the coordinates and their associated
    label (foreground or background point)

    Returns
        - coordinates [BxNx2]
        - labels [BxN]
    '''
    N = 0
    prompts = []
    for path in csv_paths:
        centroids = pd.read_csv(path).iloc[:, 1:3]
        # keep track of longest prompt
        N = len(centroids) if len(centroids) > N else N
        prompts.append(torch.tensor(centroids.values))
    # create labels: actual coords = 1 for foreground point; padding = -1
    # see https://github.com/facebookresearch/segment-anything/issues/394
    labels = [torch.ones_like(p[:,0]) for p in prompts]
    labels = [F.pad(l, pad=(0,N-l.shape[0]), value=-1) for l in labels]
    labels = torch.stack(labels).to(device)
    # pad prompts to stack them in a tensor of size BxNx2
    prompts = torch.stack([F.pad(p, pad=(0,0,0,N-p.shape[0])) for p in prompts]).to(device)

    return prompts, labels

def index_bids_dataset(datapath):
    '''
    Index an arbitrary BIDS dataset and return a data dictionary containing all images, 
    segmentations and pixel sizes.
    '''
    datapath = Path(datapath)

    # init data_dict by reading 'samples.tsv'
    samples = pd.read_csv(datapath / 'samples.tsv', delimiter='\t')
    data_dict = {}
    for i, row in samples.iterrows():
        subject = row['participant_id']
        sample = row['sample_id']
        if subject not in data_dict:
            data_dict[subject] = {}
        data_dict[subject][sample] = {}

    # populate data_dict
    sample_count = 0
    for subject in data_dict.keys():
        samples = data_dict[subject].keys()
        im_path = datapath / subject / 'micr'
        segs_path = datapath / 'derivatives' / 'labels' / subject / 'micr'
        
        images = list(im_path.glob('*.png'))
        axon_segs = list(segs_path.glob('*_seg-axon-*'))
        myelin_segs = list(segs_path.glob('*_seg-myelin-*'))
        for sample in samples:
            for img in images:
                if sample in str(img):
                    data_dict[subject][sample]['image'] = str(img)
                    sample_count += 1
            for axon_seg in axon_segs:
                if sample in str(axon_seg):
                    data_dict[subject][sample]['axon'] = str(axon_seg)
            for myelin_seg in myelin_segs:
                if sample in str(myelin_seg):
                    data_dict[subject][sample]['myelin'] = str(myelin_seg)
        # add pixel_size (assuming isotropic px size)
        json_sidecar = next((datapath / subject / 'micr').glob('*.json'))
        with open(json_sidecar, 'r') as f:
            sidecar = json.load(f)
        data_dict[subject]['px_size'] = sidecar["PixelSize"][0]
    print(f'{sample_count} samples collected.')

    return data_dict

# data loader providing the myelin map (masks), the bboxes (prompts) 
# and the path to the pre-computed image embeddings
def bids_dataloader(data_dict, maps_path, embeddings_path, sub_list):
    '''
    :param data_dict:       contains img, mask and px_size info per sample per subject
    :param maps_path:       paths to myelin maps (instance masks)
    :param embeddings_path  paths to pre-computed image embeddings
    :param sub_list         subjects included
    '''
    subjects = list(data_dict.keys())
    for sub in subjects:
        if sub in sub_list:
            samples = (s for s in data_dict[sub].keys() if 'sample' in s)
            for sample in samples:
                emb_path = embeddings_path / sub / 'micr' / f'{sub}_{sample}_TEM_embedding.pt'
                bboxes = get_sample_bboxes(sub, sample, maps_path)
                myelin_map = get_myelin_map(sub, sample, maps_path)
                yield (emb_path, bboxes, myelin_map)

class AxonDataset(Dataset):
    '''Dataset class for axon training
    This will return a resized image and an unresized ground truth. After 
    inference, the mask can be resized with sam_model.postprocess_masks given
    the original image size so that it matches the GT shape.
    '''
    def __init__(self, data_root):
        '''
        Expected data structure
        data_root/
        ├─ imgs/
        │  ├─ img1.png
        │  ├─ img2.png
        │  ├─ ...
        ├─ gts/
        │  ├─ img1_seg-axon-manual.png
        │  ├─ img2_seg-axon-manual.png
        │  ├─ ...
        '''
        self.data_root = Path(data_root)
        self.img_path = self.data_root / 'imgs'
        self.gt_path = self.data_root / 'gts'
        self.file_paths = sorted(list(self.img_path.glob("*.png")))

        self.transform = ResizeLongestSide(1024)
        print(f"number of images: {len(self.file_paths)}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        '''
        Returns a tuple containing the following:
            - image [3xHxW]
            - gt [1xHxW]
            - original size [1x2]
            - filename
        '''
        # load image and corresponding gt
        img_fname = self.file_paths[index]
        gt_fname = img_fname.name.replace('TEM.png', 'TEM_seg-axon-manual.png')
        gt_fname = self.gt_path / gt_fname
        img = cv2.imread(str(img_fname), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(gt_fname), cv2.IMREAD_GRAYSCALE)
        # assert img and gt initially have same dimensions
        assert img.shape == gt.shape, "image and ground truth should have the same size"
        original_size = img.shape
        # NOTE: we only resize the image; GT is kept at original size
        img_1024 = self.transform.apply_image(img)
        # convert shape to channel-first (in our case expand first dim to 3 channels)
        img_1024 = np.broadcast_to(img_1024, (3, img_1024.shape[0], img_1024.shape[1]))
        gt = gt[None, :, :]
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt).long(),
            torch.tensor(original_size),
            str(img_fname)
        )