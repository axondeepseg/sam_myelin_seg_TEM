#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import threshold, normalize
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import monai

import sys

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from utils import bids_utils

torch.manual_seed(444)

# datapath = Path('/home/GRAMES.POLYMTL.CA/arcol/data_axondeepseg_tem')
# derivatives_path = Path('/home/GRAMES.POLYMTL.CA/arcol/collin_project/scripts/derivatives')
# preprocessed_datapath = '/home/GRAMES.POLYMTL.CA/arcol/sam_myelin_seg_TEM/scripts/tem_split_full/train/'
# val_preprocessed_datapath = '/home/GRAMES.POLYMTL.CA/arcol/sam_myelin_seg_TEM/scripts/tem_split_full/val/'
# checkpoint = '/home/GRAMES.POLYMTL.CA/arcol/collin_project/scripts/sam_vit_b_01ec64.pth'
# device = 'cuda:0'
datapath = Path('/home/herman/Documents/NEUROPOLY_21/datasets/data_axondeepseg_tem/')
derivatives_path = Path('/home/herman/Documents/NEUROPOLY_22/COURS_MAITRISE/GBM6953EE_brainhacks_school/collin_project/scripts/derivatives/')
preprocessed_datapath = '/home/herman/Documents/NEUROPOLY_23/20230512_SAM/sam_myelin_seg_TEM/scripts/tem_split_full/train/'
val_preprocessed_datapath = '/home/herman/Documents/NEUROPOLY_23/20230512_SAM/sam_myelin_seg_TEM/scripts/tem_split_full/val/'
checkpoint = '/home/herman/Documents/NEUROPOLY_22/COURS_MAITRISE/GBM6953EE_brainhacks_school/collin_project/scripts//sam_vit_b_01ec64.pth'
device = 'cpu'
model_type = 'vit_b'

data_dict = bids_utils.index_bids_dataset(datapath)
embeddings_path = derivatives_path / 'embeddings'
maps_path = derivatives_path / 'maps'


# some utility functions to read prompts and labels
def get_myelin_bbox(bbox_df, axon_id):
    return np.array(bbox_df.iloc[axon_id])
    
def get_myelin_mask(myelin_map, axon_id):
    return 255 * (myelin_map == axon_id + 1)


# Load the initial model checkpoint
sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model.to(device)
params = list(sam_model.image_encoder.parameters()) + list(sam_model.mask_decoder.parameters())


# utility function to segment the whole image without the SamPredictor class
# TODO: RE-WRITE THIS
def segment_image(sam_model, imgs, prompts, device):
    
    full_mask = None
    
    input_size = imgs.shape
    imgs = sam_model.preprocess(imgs.to(device))
    image_embedding = sam_model.image_encoder(imgs)
    
    prompts = bids_utils.PromptSet(prompts.squeeze())
    prompt_loader = DataLoader(prompts, batch_size=prompt_batch_size)
    
    for axon_id, bboxes in prompt_loader:
        if np.isnan(prompt).any():
                continue
        
        with torch.no_grad():
            
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=bboxes.to(device),
                masks=None,
            )
            
            low_res_mask, _ = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            mask = sam_model.postprocess_masks(
                low_res_mask,
                input_size,
                original_size,
            ).to(device)
            binary_mask = normalize(threshold(mask, 0.0, 0))

        if full_mask is None:
            full_mask = binary_mask
        else:
            full_mask += binary_mask

        # TODO binarize final mask (overlapping regions end up with values >1)    
    return full_mask

# Training hyperparameters
lr = 1e-6
wd = 0.01
optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
loss_fn = monai.losses.DiceLoss(sigmoid=True)
num_epochs = 100
val_frequency = 4
batch_size = 1
prompt_batch_size = 10
mean_epoch_losses = []
transform = ResizeLongestSide(sam_model.image_encoder.img_size)
run_id = 'run1'

# loaders
train_dset = bids_utils.MyelinDataset(preprocessed_datapath)
val_dset = bids_utils.MyelinDataset(val_preprocessed_datapath)

train_dataloader = DataLoader(
    train_dset,
    batch_size=batch_size,
    shuffle=True,
)
val_loader = DataLoader(val_dset, batch_size=1)

best_val_loss = 1000
best_val_epoch = -1

for epoch in range(num_epochs):
    epoch_losses = []
    val_losses = []
    
    sam_model.train()
    for (imgs, gts, prompts, sizes, names) in train_dataloader:

        # IMG ENCODER
        input_size = imgs.shape
        imgs = sam_model.preprocess(imgs.to(device))
        image_embedding = sam_model.image_encoder(imgs)
        
        # batch and shuffle prompts
        prompt_loader = DataLoader(
            bids_utils.PromptSet(prompts.squeeze()),
            batch_size=prompt_batch_size,
            shuffle=True
        )
        # train on every axon in the image
        for axon_ids, prompts in prompt_loader:
            # build mask by summing individual masks at train-time
            # this mask will contain prompt_batch_size myelin sheaths
            labels = torch.zeros_like(gts)
            for b in range(batch_size):
                mask = gts[b]
                individual_masks = [get_myelin_mask(mask, a_id) for a_id in axon_ids]
                individual_masks = torch.stack(individual_masks)
                labels[b] = torch.sum(individual_masks, dim=0)

            # empty masks should not be processed
            if np.isnan(prompts).any():
                continue
            
            # no grad for the prompt encoder
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=prompts.to(device),
                    masks=None,
                )
            # now we pass the image and prompt embeddings in the mask decoder
            low_res_mask, _ = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            upscaled_mask = sam_model.postprocess_masks(
                low_res_mask,
                input_size=input_size,
                original_size=(sizes[0][0], sizes[0][1]),
            ).to(device)
            
            gt_binary_mask = torch.as_tensor(labels.to(device) > 0, dtype=torch.float32)
            
            optimizer.zero_grad()
            loss = loss_fn(upscaled_mask, gt_binary_mask)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
    
    # validation loop every 5 epochs to avoid cluttering
    if epoch % val_frequency == 0:
        mean_val_loss = 0

        sam_model.eval()
        with torch.no_grad():
            #TODO: VALIDATION LOOP (could use segment_image fn)
            pass
            # for sample in val_dataloader:
            #     emb_path, bboxes, myelin_map = sample
            #     emb_dict = load_image_embedding(emb_path)
            #     mask = segment_image(sam_model, bboxes, emb_dict, device)

        if mean_val_loss < best_val_loss:
            print("\tSaving best model.")
            best_val_loss = mean_val_loss
            best_val_epoch = epoch
            torch.save(sam_model.state_dict(), f'sam_vit_b_01ec64_myelin-seg_{run_id}_best.pth')
    mean_epoch_losses.append(np.mean(epoch_losses))
    print(f'EPOCH {epoch} MEAN LOSS: {mean_epoch_losses[-1]}')
torch.save(sam_model.state_dict(), f'sam_vit_b_01ec64_myelin-seg_{run_id}_final.pth')

# Plot mean epoch losses

plt.plot(list(range(len(mean_epoch_losses))), mean_epoch_losses)
plt.title('Mean epoch loss')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')

plt.savefig(f'losses_myelin_seg_vit_b_{run_id}.png')
