# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
import colorsys
import cv2
import torch.nn.functional as F


def calculate_iou(prediction, mask):
    intersection = prediction * mask
    union = prediction + mask - intersection
    return intersection.sum() / (union.sum() + 1e-7)


def get_crops_coords(image_size, patch_size, num_patchs_per_side):
    h, w = image_size
    if num_patchs_per_side == 1:
        x_step_size = y_step_size = 0
    else:
        x_step_size = (w - patch_size) // (num_patchs_per_side - 1)
        y_step_size = (h - patch_size) // (num_patchs_per_side - 1)
    crops_coords = []
    for i in range(num_patchs_per_side):
        for j in range(num_patchs_per_side):
            y_start, y_end, x_start, x_end = (
                i * y_step_size,
                i * y_step_size + patch_size,
                j * x_step_size,
                j * x_step_size + patch_size,
            )
            crops_coords.append([y_start, y_end, x_start, x_end])
    return crops_coords

def get_random_crop_coordinates(crop_scale_range, image_width, image_height):
    rand_number = random.random()
    rand_number *= crop_scale_range[1] - crop_scale_range[0]
    rand_number += crop_scale_range[0]
    patch_size = int(rand_number * min(image_width, image_height))
    if patch_size != min(image_width, image_height):
        x_start = random.randint(0, image_width - patch_size)
        y_start = random.randint(0, image_height - patch_size)
    else:
        x_start = 0
        y_start = 0
    return x_start, x_start + patch_size, y_start, y_start + patch_size


def generate_distinct_colors(n):
    colors = []
    if n == 1:
        return [(255, 255, 255)]
    for i in range(n):
        hue = i / n
        saturation = 0.9
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        scaled_rgb = tuple(int(x * 255) for x in rgb)
        colors.append(scaled_rgb)
    return colors


def get_boundry_and_eroded_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    open_mask = np.zeros_like(mask)
    #eroded_mask = np.zeros_like(mask)
    #boundry_mask = np.zeros_like(mask)
    for part_mask_idx in np.unique(mask)[1:]:
        part_mask = np.where(mask == part_mask_idx, 1, 0)
        #part_mask_erosion = cv2.erode(part_mask.astype(np.uint8), kernel, iterations=1)
        part_mask_open = cv2.morphologyEx(part_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        #part_boundry_mask = part_mask - part_mask_erosion
        #eroded_mask = np.where(part_mask_erosion > 0, part_mask_idx, eroded_mask)
        open_mask = np.where(part_mask_open > 0, part_mask_idx, open_mask)
        #boundry_mask = np.where(part_boundry_mask > 0, part_mask_idx, boundry_mask)
    #return eroded_mask, boundry_mask
    return open_mask

def get_colored_segmentation(mask, boundry_mask, image):
    color = (255, 112, 132)
    
    boundry_mask_rgb = 0
    if boundry_mask is not None:
        boundry_mask_rgb = torch.repeat_interleave(boundry_mask[None, ...], 3, 0).type(torch.float)
        for j in range(3):
            boundry_mask_rgb[j] = torch.where(
                boundry_mask_rgb[j] == 1, 
                color[j] / 255, 
                boundry_mask_rgb[j]
            )
    
    mask_rgb = torch.repeat_interleave(mask[None, ...], 3, 0).type(torch.float)
    for j in range(3):
        mask_rgb[j] = torch.where(
            mask_rgb[j] == 1, 
            color[j] / 255, 
            mask_rgb[j]
        )

    if boundry_mask is not None:
        return (boundry_mask_rgb * 0.6 + mask_rgb * 0.6 + image * 0.4).permute(1, 2, 0)
    else:
        return (mask_rgb * 0.6 + image * 0.4).permute(1, 2, 0)

def create_pseudo_image(image, mask):
    if mask.dim() == 2:  
        mask = mask.unsqueeze(0).unsqueeze(0)  
    elif mask.dim() == 3:  
        mask = mask.unsqueeze(1)  
    elif mask.size(1) != 1:
        raise ValueError("The mask should have a single channel dimension.")

   
    batch_size, _, height, width = mask.shape

    
    pseudo_mask = torch.zeros((batch_size, 3, height, width), device=mask.device, dtype=mask.dtype)

   
    pseudo_mask[:, 0] = torch.where(mask[:, 0] == 0, torch.tensor(1, device=mask.device, dtype=mask.dtype), torch.tensor(127, device=mask.device, dtype=mask.dtype))
    pseudo_mask[:, 1] = torch.where(mask[:, 0] == 0, torch.tensor(127, device=mask.device, dtype=mask.dtype), torch.tensor(1, device=mask.device, dtype=mask.dtype))
    pseudo_mask[:, 2] = torch.where(mask[:, 0] == 0, torch.tensor(64, device=mask.device, dtype=mask.dtype), torch.tensor(64, device=mask.device, dtype=mask.dtype))
    
    
    #pseudo_mask = torch.where(mask == 1, image, pseudo_mask)
    

    
    pseudo_mask = F.interpolate(pseudo_mask, size=(512, 512), mode="bilinear")

   
    pseudo_image = (pseudo_mask + (image * 255) / 3).clamp(0, 255).byte()
    
    #pseudo_image = pseudo_mask.clamp(0, 255).byte()

    
    return pseudo_image / 255


def get_mask(pseudo_image, image):
    pseudo_mask = pseudo_image * 255 - F.interpolate(
        (image * 255) / 3, size=(512, 512), mode='bilinear', align_corners=False
    )
    
    #pseudo_mask = pseudo_image * 255
    
    final_mask = torch.where(
        pseudo_mask[:, 0, :, :] < pseudo_mask[:, 1, :, :],
        torch.zeros_like(pseudo_mask[:, 0, :, :]),  
        torch.ones_like(pseudo_mask[:, 0, :, :]) 
    )
    
    
    #avg_values = pseudo_mask.mean(dim=1, keepdim=True)  # shape: (b, 1, h, w)
    
    
    #max_h = avg_values.max(dim=2, keepdim=True)[0]
    
    #max_avg = max_h.max(dim=3, keepdim=True)[0]  
    
   
    #final_mask = (avg_values > 0).float().squeeze(1)  
    #final_mask = (pseudo_mask > 128).all(dim=1).float()  

    return final_mask


def fill_holes_and_remove_small_areas(mask, min_hole_size=500, min_area_size=1000):

    if mask.is_cuda:
        mask = mask.cpu().numpy().astype(np.uint8)

   
    inverted_mask = 1 - mask

   
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted_mask, connectivity=8)

    for i in range(1, num_labels):  
        if stats[i, cv2.CC_STAT_AREA] < min_hole_size:
            inverted_mask[labels == i] = 0  

   
    mask = 1 - inverted_mask

    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    
    new_mask = np.zeros_like(mask)

    for i in range(1, num_labels):  
        if stats[i, cv2.CC_STAT_AREA] >= min_area_size:
            new_mask[labels == i] = 1 

    
    new_mask = torch.tensor(new_mask, dtype=torch.uint8)
    if torch.cuda.is_available():
        new_mask = new_mask.cuda()

    return new_mask


