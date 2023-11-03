#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch.autograd import Variable as V
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
torch.autograd.set_detect_anomaly(True)

datapath = Path('/home/GRAMES.POLYMTL.CA/arcol/data_axondeepseg_tem')
derivatives_path = Path('/home/GRAMES.POLYMTL.CA/arcol/collin_project/scripts/derivatives')
preprocessed_datapath = '/home/GRAMES.POLYMTL.CA/arcol/sam_myelin_seg_TEM/scripts/tem_split_full/train/'
val_preprocessed_datapath = '/home/GRAMES.POLYMTL.CA/arcol/sam_myelin_seg_TEM/scripts/tem_split_full/val/'
checkpoint = '/home/GRAMES.POLYMTL.CA/arcol/sam_myelin_seg_TEM/scripts/sam_vit_b_01ec64.pth'
device = 'cuda:0'
# datapath = Path('/home/herman/Documents/NEUROPOLY_21/datasets/data_axondeepseg_tem/')
# derivatives_path = Path('/home/herman/Documents/NEUROPOLY_22/COURS_MAITRISE/GBM6953EE_brainhacks_school/collin_project/scripts/derivatives/')
# preprocessed_datapath = '/home/herman/Documents/NEUROPOLY_23/20230512_SAM/sam_myelin_seg_TEM/scripts/tem_split_full/train/'
# val_preprocessed_datapath = '/home/herman/Documents/NEUROPOLY_23/20230512_SAM/sam_myelin_seg_TEM/scripts/tem_split_full/val/'
# checkpoint = '/home/herman/Documents/NEUROPOLY_22/COURS_MAITRISE/GBM6953EE_brainhacks_school/collin_project/scripts/sam_vit_b_01ec64.pth'
# device = 'cpu'

print(f'current memory allocated (before everything): {torch.cuda.memory_allocated}')

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
# params = list(sam_model.image_encoder.parameters()) + list(sam_model.mask_decoder.parameters())


# utility function to segment the whole image without the SamPredictor class
def segment_image(sam_model, imgs, prompts, original_size, device):
    
    full_mask = None
    input_size = imgs.shape
    imgs = sam_model.preprocess(imgs.to(device))
    
    prompts = bids_utils.PromptSet(prompts.squeeze())
    prompt_loader = DataLoader(prompts, batch_size=prompt_batch_size)
    
    for _, bboxes in prompt_loader:
        if np.isnan(bboxes).any():
                continue
        
        with torch.no_grad():
            image_embedding = sam_model.image_encoder(imgs)
            
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
                (input_size[-2], input_size[-1]),
                original_size,
            ).to(device)
            combined_mask = torch.sum(torch.sigmoid(mask), dim=0)

        if full_mask is None:
            full_mask = combined_mask
        else:
            full_mask += combined_mask
    return full_mask[None, :]

def jitter_and_clamp(prompts, j_range, max_size):
    'Jitter prompt coordinates and clamp them to [0, max_size-1] range.'
    x0_jit = torch.randint_like(prompts[:, :, 1], low=-j_range, high=j_range)
    y0_jit = torch.randint_like(prompts[:, :, 2], low=-j_range, high=j_range)
    x1_jit = torch.randint_like(prompts[:, :, 3], low=-j_range, high=j_range)
    y1_jit = torch.randint_like(prompts[:, :, 4], low=-j_range, high=j_range)
    prompts[:, :, 1] = torch.clamp(prompts[:, :, 1] + x0_jit, min=0, max=max_size[0]-1)
    prompts[:, :, 2] = torch.clamp(prompts[:, :, 2] + y0_jit, min=0, max=max_size[1]-1)
    prompts[:, :, 3] = torch.clamp(prompts[:, :, 3] + x1_jit, min=0, max=max_size[0]-1)
    prompts[:, :, 4] = torch.clamp(prompts[:, :, 4] + y1_jit, min=0, max=max_size[1]-1)
    return prompts

# Training hyperparameters
lr = 1e-6
wd = 0.01
optimizer = torch.optim.AdamW(
    params=list(sam_model.image_encoder.parameters()) + list(sam_model.mask_decoder.parameters()), 
    lr=lr, 
    weight_decay=wd
)
loss_fn = monai.losses.DiceLoss(sigmoid=True)
val_loss_fn = monai.losses.DiceLoss(sigmoid=False)
num_epochs = 100
val_frequency = 4
batch_size = 1
prompt_batch_size = 10
use_full_prompt_batch_size = True
mean_epoch_losses = []
mean_val_losses = []
val_epochs = []
transform = ResizeLongestSide(sam_model.image_encoder.img_size)
jitter_coords = True
jitter_range = 20
run_id = 'run4'

# loaders
train_dset = bids_utils.MyelinDataset(preprocessed_datapath)
val_dset = bids_utils.MyelinDataset(val_preprocessed_datapath)

train_loader = DataLoader(
    train_dset,
    batch_size=batch_size,
    shuffle=True,
    device=device
)
val_loader = DataLoader(val_dset, batch_size=batch_size, device=device)

best_val_loss = 1000
best_val_epoch = -1

print(f'current memory allocated (before training loop): {torch.cuda.memory_allocated}')

for epoch in range(num_epochs):
    epoch_losses = []
    val_losses = []
    
    sam_model.train()
    for (imgs, gts, prompts, sizes, names) in tqdm(train_loader):
        # IMG ENCODER
        print(f'current memory allocated (start of training loop): {torch.cuda.memory_allocated}')

        input_size = imgs.shape
        imgs = sam_model.preprocess(imgs.to(device))
        image_embedding = sam_model.image_encoder(imgs)

        print(f'current memory allocated (before loading bboxes): {torch.cuda.memory_allocated}')


        if jitter_coords:
            prompts = jitter_and_clamp(prompts, jitter_range, input_size[-2:])
        prompt_dataset = bids_utils.PromptSet(prompts.squeeze())
        if use_full_prompt_batch_size:
            prompt_batch_size = len(prompt_dataset)
        # batch and shuffle prompts
        prompt_loader = DataLoader(
            prompt_dataset,
            batch_size=prompt_batch_size,
            shuffle=True
        )
        # train on every axon in the image
        for axon_ids, prompts in prompt_loader:
            # build mask by stacking individual masks at train-time
            # this mask will contain prompt_batch_size myelin sheaths
            labels = torch.zeros_like(gts)
            # TODO: currently doesnt work for batch_size >1
            for b in range(batch_size):
                mask = gts[b]
                individual_masks = [get_myelin_mask(mask, a_id) for a_id in axon_ids]
                individual_masks = torch.stack(individual_masks)
                # labels[b] = torch.sum(individual_masks, dim=0)
                # TODO: the GTs were summed but SAM outputs prompt_batch_size number of stacked masks...
                labels = individual_masks

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
                input_size=(input_size[-2], input_size[-1]),
                original_size=(sizes[0][0], sizes[0][1]),
            ).to(device)
            # upscaled_mask = torch.sum(upscaled_mask, dim=0)
            
            gt_binary_mask = torch.as_tensor(labels.to(device) > 0, dtype=torch.float32)

            loss = loss_fn(upscaled_mask, gt_binary_mask)

            loss.backward()
            epoch_losses.append(loss.item())
        
            optimizer.step()
            optimizer.zero_grad()

    # validation loop
    if epoch % val_frequency == 0:
        mean_val_loss = 0
        val_epochs.append(epoch)

        sam_model.eval()
        with torch.no_grad():
            for v_imgs, v_gts, v_prompts, v_sizes, _ in val_loader:
                v_sizes = (v_sizes[0][0], v_sizes[0][1])
                mask = segment_image(sam_model, v_imgs, v_prompts, v_sizes, device)
                gt_binary_mask = torch.as_tensor(v_gts > 0, dtype=torch.float32)
                v_loss = val_loss_fn(mask, gt_binary_mask.to(device))
                val_losses.append(v_loss.item())
        mean_val_loss = np.mean(val_losses)

        mean_val_losses.append(mean_val_loss)
        print(f'EPOCH {epoch}\n\tMEAN VAL LOSS: {mean_val_losses[-1]}')

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
plt.plot(val_epochs, mean_val_losses)
plt.legend(['Training loss', 'Validation loss'])
plt.title('Mean epoch loss for myelin segmentation')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')

plt.savefig(f'losses_myelin_seg_vit_b_{run_id}.png')
