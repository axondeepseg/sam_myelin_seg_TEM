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
def segment_image(sam_model, bboxes, emb_dict, device):
    
    original_size = emb_dict['original_size']
    input_size = emb_dict['input_size']
    image_embedding = emb_dict['features']
    full_mask = None

    for axon_id in range(len(bboxes)):
        prompt = get_myelin_bbox(bboxes, axon_id)
        if np.isnan(prompt).any():
                continue
        
        with torch.no_grad():
            box = transform.apply_boxes(prompt, original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            box_torch = box_torch[None, :]

            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
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
        # image_embedding = sam_model.image_encoder(imgs)
        
        # batch and shuffle prompts
        prompt_loader = DataLoader(
            bids_utils.PromptSet(prompts.squeeze()),
            batch_size=prompt_batch_size,
            shuffle=True
        )
        print(names)
        # train on every axon in the image
        for axon_ids, prompts in prompt_loader:
            # get mask and bbox prompt
            print(axon_ids)
            print(prompts)
            continue
            prompt = get_myelin_bbox(bboxes, axon_id)
            gt_mask = get_myelin_mask(myelin_map, axon_id)

            # empty masks should not be processed
            if np.isnan(prompt).any():
                continue
            # no grad for the prompt encoder
            with torch.no_grad():
                box = transform.apply_boxes(prompt, original_size)
                box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
                box_torch = box_torch[None, :]
                
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=box_torch,
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
                input_size,
                original_size,
            ).to(device)
            
            gt_mask_resized = torch.from_numpy(gt_mask[:,:,0]).unsqueeze(0).unsqueeze(0).to(device)
            gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)
            
            loss = loss_fn(upscaled_mask, gt_binary_mask)
            loss.backward()
            pbar.set_description(f'Loss: {loss.item()}')
            epoch_losses.append(loss.item())
        # step the optimizer
        optimizer.step()
        optimizer.zero_grad()
        pbar.update(1)
    
    # validation loop every 5 epochs to avoid cluttering
    if epoch % val_frequency == 0:
        sam_model.eval()
        with torch.no_grad():
            for sample in val_dataloader:
                emb_path, bboxes, myelin_map = sample
                emb_dict = load_image_embedding(emb_path)
                mask = segment_image(sam_model, bboxes, emb_dict, device)
                #TODO: compute loss and save best model
                # fname = emb_path.stem.replace('embedding', f'val-seg-epoch{epoch}.png')
                # plt.imsave(Path('validation_results') / fname, mask.cpu().detach().numpy().squeeze(), cmap='gray')

    # if epoch % 10 == 0:
    #     torch.save(sam_model.state_dict(), f'sam_vit_b_01ec64_epoch_{epoch}_diceloss.pth')
    mean_epoch_losses.append(np.mean(epoch_losses))
    print(f'EPOCH {epoch} MEAN LOSS: {mean_epoch_losses[-1]}')
torch.save(sam_model.state_dict(), f'../../scripts/sam_vit_b_01ec64_auto-myelin-seg_{run_id}_final.pth')

# Plot mean epoch losses

plt.plot(list(range(len(mean_epoch_losses))), mean_epoch_losses)
plt.title('Mean epoch loss')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')

plt.savefig('losses_with_diceloss.png')
