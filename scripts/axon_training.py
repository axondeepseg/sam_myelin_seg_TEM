#!/usr/bin/env python
# coding: utf-8

# This script launches training for automatic axon segmentation (predicts
# the full mask, unlike the myelin training pipeline)

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import monai
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from utils import bids_utils


datapath = Path('/home/GRAMES.POLYMTL.CA/arcol/data_axondeepseg_tem')
derivatives_path = Path('/home/GRAMES.POLYMTL.CA/arcol/collin_project/scripts/derivatives')
checkpoint = '/home/GRAMES.POLYMTL.CA/arcol/sam_myelin_seg_TEM/scripts/sam_vit_b_01ec64.pth'
device = 'cuda:0'
preprocessed_data_path = '/home/GRAMES.POLYMTL.CA/arcol/sam_myelin_seg_TEM/scripts/tem_split_full/train/'
val_preprocessed_datapath = '/home/GRAMES.POLYMTL.CA/arcol/sam_myelin_seg_TEM/scripts/tem_split_full/val/'
# datapath = Path('/home/herman/Documents/NEUROPOLY_21/datasets/data_axondeepseg_tem/')
# derivatives_path = Path('/home/herman/Documents/NEUROPOLY_22/COURS_MAITRISE/GBM6953EE_brainhacks_school/collin_project/scripts/derivatives/')
# checkpoint = '/home/herman/Documents/NEUROPOLY_22/COURS_MAITRISE/GBM6953EE_brainhacks_school/collin_project/scripts//sam_vit_b_01ec64.pth'
# device = 'cpu'
# preprocessed_data_path = '/home/herman/Documents/NEUROPOLY_23/20230512_SAM/sam_myelin_seg_TEM/scripts/tem_split_full/train/'
# val_preprocessed_datapath = '/home/herman/Documents/NEUROPOLY_23/20230512_SAM/sam_myelin_seg_TEM/scripts/tem_split_full/val/'


data_dict = bids_utils.index_bids_dataset(datapath)
maps_path = derivatives_path / 'maps'

# helper functions to display masks/bboxes
def show_mask(mask, ax):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def load_centroid_prompts(csv_paths, device):
    N = 0
    prompts = []
    for path in csv_paths:
        centroids = pd.read_csv(path).iloc[:, 1:3]
        N = len(centroids) if len(centroids) > N else N
        prompts.append(torch.tensor(centroids.values))
    # create labels: actual coords = 1 for foreground point; padding = -1
    labels = [torch.ones_like(p[:,0]) for p in prompts]
    labels = [F.pad(l, pad=(0,N-l.shape[0]), value=-1) for l in labels]
    labels = torch.stack(labels).to(device)
    # pad prompts
    prompts = torch.stack([F.pad(p, pad=(0,0,0,N-p.shape[0])) for p in prompts]).to(device)

    return prompts, labels

# Load the initial model checkpoint
model_type = 'vit_b'

sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model.to(device)
sam_model.train()

def load_image_embedding(path):
    emb_dict = torch.load(path, device)
    return emb_dict

# Training hyperparameters
lr = 1e-4
wd = 0.01
optimizer = torch.optim.AdamW(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)
loss_fn = monai.losses.DiceLoss(sigmoid=True)


# Training loop
num_epochs = 100
batch_size = 4
mean_epoch_losses = []
mean_val_losses = []
val_epochs = []
val_frequency = 4
prompt_with_centroids = True
run_id='run5'

transform = ResizeLongestSide(sam_model.image_encoder.img_size)
train_dset = bids_utils.AxonDataset(preprocessed_data_path)
train_dataloader = DataLoader(
    train_dset,
    batch_size=batch_size,
    shuffle=True,
)

val_dset = bids_utils.AxonDataset(val_preprocessed_datapath)
val_dataloader = DataLoader(val_dset, batch_size=1)
best_val_loss = 1_000
best_val_epoch = -1

for epoch in range(num_epochs):
    epoch_losses = []
    val_losses = []

    sam_model.train()
    for (imgs, gts, sizes, names) in tqdm(train_dataloader):
        
        # IMAGE ENCODER
        input_size = imgs.shape
        imgs = sam_model.preprocess(imgs.to(device))
        image_embedding = sam_model.image_encoder(imgs)
        
        # PROMPT ENCODER
        with torch.no_grad():
            
            if prompt_with_centroids:
                names = [Path(n).name for n in names]
                prompt_paths = [maps_path / n.split('_')[0] / 'micr' / n for n in names]
                prompt_paths = [str(p).replace('_TEM.png', '_prompts.csv') for p in prompt_paths]
                prompts, labels = load_centroid_prompts(prompt_paths, device)
                prompts = transform.apply_coords_torch(prompts, (sizes[0][0], sizes[0][1]))
                # note: for this to work, might need to modify SAM source files;
                # see https://github.com/facebookresearch/segment-anything/issues/365
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=(prompts, labels),
                    boxes=None,
                    masks=None,
                )
            else:
                # use full bbox
                H, W = torch.tensor(input_size[-2]), torch.tensor(input_size[-1])
                boxes = torch.stack([
                    torch.zeros_like(H),
                    torch.zeros_like(H),
                    W-1,
                    H-1
                ]).t()[None, :]
                box_torch = torch.as_tensor(boxes, dtype=torch.float, device=device)
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None,
                )

        # MASK DECODER
        low_res_mask, _, = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        upscaled_mask = sam_model.postprocess_masks(
            low_res_mask,
            input_size=input_size[-2:],
            original_size=(sizes[0][0], sizes[0][1]),
        ).to(device)

        gt_mask_resized = gts.to(device)
        gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)
            
        loss = loss_fn(upscaled_mask, gt_binary_mask)
        loss.backward()
        epoch_losses.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
    
    # validation loop every 5 epochs to avoid cluttering
    #TODO validation loop needs to be updated
    if epoch % val_frequency == 0:
        sam_model.eval()
        val_epochs.append(epoch)
        for (val_imgs, val_gts, val_sizes, val_names) in val_dataloader:
            with torch.no_grad():
                input_size = val_imgs.shape
                val_imgs = sam_model.preprocess(val_imgs.to(device))
                image_embedding = sam_model.image_encoder(val_imgs)
                
                if prompt_with_centroids:
                    val_names = [Path(n).name for n in val_names]
                    val_prompt_paths = [maps_path / n.split('_')[0] / 'micr' / n for n in val_names]
                    val_prompt_paths = [str(p).replace('_TEM.png', '_prompts.csv') for p in val_prompt_paths]
                    val_prompts, val_labels = load_centroid_prompts(val_prompt_paths, device)
                    val_prompts = transform.apply_coords_torch(val_prompts, (val_sizes[0][0], val_sizes[0][1]))
                    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                        points=(val_prompts, val_labels),
                        boxes=None,
                        masks=None,
                    )
                else:
                    H, W = torch.tensor(input_size[-2]), torch.tensor(input_size[-1])
                    val_boxes = torch.stack([torch.zeros_like(H), torch.zeros_like(H), W-1, H-1]).t()[None, :]
                    val_boxes = torch.as_tensor(val_boxes, dtype=torch.float, device=device)
                    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                        points=None,
                        boxes=val_boxes,
                        masks=None,
                    )  
                
                low_res_mask, _ = sam_model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                upscaled_mask = sam_model.postprocess_masks(
                    low_res_mask,
                    input_size=input_size[-2:],
                    original_size=(sizes[0][0], sizes[0][1]),
                ).to(device)

            val_gts = val_gts.to(device)
            gt_binary_mask = torch.as_tensor(val_gts > 0, dtype=torch.float32)

            val_loss = loss_fn(upscaled_mask, gt_binary_mask)
            val_losses.append(val_loss.item())

        mean_val_loss = np.mean(val_losses)
        mean_val_losses.append(mean_val_loss)
        print(f'EPOCH {epoch}\n\tMEAN VAL LOSS: {mean_val_losses[-1]}')
        if mean_val_loss < best_val_loss:
            print("\tSaving best model.")
            best_val_loss = mean_val_loss
            best_val_epoch = epoch
            torch.save(sam_model.state_dict(), f'sam_vit_b_01ec64_auto-axon-seg_{run_id}_best.pth')
    # TODO: save validation img
    # fname = emb_path.stem.replace('embedding', f'val-seg-axon_epoch{epoch}.png')
    # plt.imsave(Path('axon_validation_results') / fname, binary_mask.cpu().detach().numpy().squeeze(), cmap='gray')

    mean_epoch_losses.append(np.mean(epoch_losses))
    print(f'EPOCH {epoch}\n\tMEAN LOSS: {mean_epoch_losses[-1]}')

print(f'Best model checkpoint was saved at epoch {best_val_epoch}.')
# save final checkpoint
torch.save(sam_model.state_dict(), f'sam_vit_b_01ec64_auto-axon-seg_{run_id}_final.pth')

# Plot mean epoch losses

plt.plot(list(range(len(mean_epoch_losses))), mean_epoch_losses)
print(val_epochs)
print(mean_val_losses)
plt.plot(val_epochs, mean_val_losses)
plt.legend(['Training loss', 'Validation loss'])
plt.title('Mean epoch loss for axon segmentation')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.savefig(f'losses_axon_seg_vit_b_{run_id}.png')
