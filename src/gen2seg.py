# -*- coding: utf-8 -*-
import os

import pytorch_lightning as pl
import torch
from torch import optim
import torch.nn.functional as F

from src.stablediffusion import StableDiffusion
from src.utils import (
    calculate_iou,
    get_crops_coords,
    generate_distinct_colors,
    get_colored_segmentation,
    get_boundry_and_eroded_mask,
    create_pseudo_image,
    get_mask,
    fill_holes_and_remove_small_areas,
)
import gc
from PIL import Image
import numpy as np

class Gen2Seg(pl.LightningModule):
    def __init__(self, config, learning_rate=0.001):
        super().__init__()
        self.counter = 0
        self.val_counter = 0
        self.config = config
        self.save_hyperparameters(config.__dict__)
        self.learning_rate = learning_rate
        self.max_val_iou = 0
        self.val_ious = []

        self.stable_diffusion = StableDiffusion(
            sd_version="2.1",
                    )

        self.checkpoint_dir = None
        if self.config.train:
            self.num_parts = len(self.config.part_names)
        else:
            self.num_parts = (
                    len(
                        [
                            file
                            for file in os.listdir(self.config.checkpoint_dir)
                            if file.endswith(".pth")
                        ]
                    )
                    #+ 1
            )
            assert (
                    self.num_parts > 0
            ), "a folder path should be passed to --checkpoints_dir, which contains the text embeddings!"

        self.prepare_text_embeddings()
        del self.stable_diffusion.tokenizer
        del self.stable_diffusion.text_encoder
        torch.cuda.empty_cache()

        self.embeddings_to_optimize = []
        if self.config.train:
            
            selected_indices = [3, 1]  

            for i in selected_indices:
                
                embedding = self.text_embedding[:, i: i + 1].clone()
                embedding.requires_grad_(True)  
                self.embeddings_to_optimize.append(embedding)

        
        self.token_ids = 2

    def prepare_text_embeddings(self):

        segment_prefix = "Segment"

        if self.config.text_prompt is None:
            if len(self.config.part_names) >= 2:
                
                text_prompt = f"pink {self.config.part_names[1]} green {self.config.part_names[0]}"
                #text_prompt = f"{self.config.part_names[1]} {self.config.part_names[0]}"
            else:
                raise ValueError("The part_names list must contain at least two elements.")
        else:
            text_prompt = self.config.text_prompt

        (
            self.uncond_embedding,
            self.text_embedding,
        ) = self.stable_diffusion.get_text_embeds(text_prompt, "")

    def on_fit_start(self) -> None:
        self.checkpoint_dir = os.path.join(
            self.config.output_dir, "checkpoints", self.logger.log_dir.split("/")[-1]
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.stable_diffusion.setup(self.device)
        self.uncond_embedding, self.text_embedding = self.uncond_embedding.to(
            self.device
        ), self.text_embedding.to(self.device)

    def training_step(self, batch, batch_idx):
        
        image, mask = batch
        

        pseudo_image = create_pseudo_image(image, mask)

        #mask = mask[0]

        text_embedding = torch.cat(
            [
                self.text_embedding[:, 0:1],  
                self.embeddings_to_optimize[1].to(self.device), 
                self.text_embedding[:, 2:3],  
                self.embeddings_to_optimize[0].to(self.device),  
                self.text_embedding[:, 4:], 
            ],
            dim=1,
        )
        
        t_embedding = torch.cat([self.uncond_embedding, text_embedding])

        sd_loss, pred_pseudo_image = self.stable_diffusion.train_step(
            t_embedding,
            image,
            pseudo_image,
            t=torch.tensor(self.config.train_t),
            #attention_output_size=128,
        )

        loss1 = F.mse_loss(pred_pseudo_image, pseudo_image)

        pred_mask = get_mask(pred_pseudo_image, image)

        one_shot_mask = mask
        #one_shot_mask = one_shot_mask.unsqueeze(0)
        
        
        loss2 = F.mse_loss(pred_mask, one_shot_mask)
        

        pred_mask = F.interpolate(pred_mask.unsqueeze(0).float(), size=(128, 128), mode='nearest').squeeze(0)
        
        one_shot_mask = F.interpolate(one_shot_mask.unsqueeze(0).float(), size=(128, 128), mode='nearest').squeeze(0)
        

        loss3 = F.mse_loss(pred_mask, one_shot_mask)

        loss = self.config.sd_loss_coef * sd_loss + self.config.pixcel_loss_coef * loss1 + self.config.mask_loss_512_coef * loss2 + self.config.mask_loss_128_coef * loss3
        

        self.test_t_embedding = t_embedding

        final_mask = self.get_patched_masks(
            image,
            self.config.train_mask_size,
        )

        ious = []
        for idx, part_name in enumerate(self.config.part_names):
            part_mask = torch.where(mask == idx, 1, 0).type(torch.uint8)
            if torch.all(part_mask == 0):
                continue
            iou = calculate_iou(
                torch.where(final_mask == idx, 1, 0).type(torch.uint8), part_mask
            )
            ious.append(iou)
            self.log(f"train {part_name} iou", iou, on_step=True, sync_dist=True)
        mean_iou = sum(ious) / len(ious)
        # print("mean_iou:", mean_iou)

        self.log("sd_loss", sd_loss.detach().cpu(), on_step=True, sync_dist=True)
        self.log("loss1", loss1.detach().cpu(), on_step=True, sync_dist=True)
        self.log("loss2", loss2.detach().cpu(), on_step=True, sync_dist=True)
        self.log("loss3", loss2.detach().cpu(), on_step=True, sync_dist=True)
        self.log("loss", loss.detach().cpu(), on_step=True, sync_dist=True)
        self.log("train mean iou", mean_iou.cpu(), on_step=True, sync_dist=True)

        return loss

    def get_patched_masks(self, image, output_size):
        crops_coords = get_crops_coords(
            image.shape[2:],
            self.config.patch_size,
            self.config.num_patchs_per_side,
        )

        
        final_pseudo_image = torch.zeros(
            image.shape[0],  
            image.shape[1],  
            output_size,
            output_size,
        ).to(self.device)

        for crop_coord in crops_coords:
            y_start, y_end, x_start, x_end = crop_coord
           
            cropped_image = image[:, :, y_start:y_end, x_start:x_end]

            with torch.no_grad():
                pseudo_image = self.stable_diffusion.inference_step(
                    self.test_t_embedding,
                    cropped_image,
                    t=torch.tensor(self.config.test_t),
                    generate_new_noise=True,
                )

            ratio = 512 // output_size
            mask_y_start, mask_y_end, mask_x_start, mask_x_end = (
                y_start // ratio,
                y_end // ratio,
                x_start // ratio,
                x_end // ratio,
            )

            pseudo_image_resized = F.interpolate(
                pseudo_image,
                size=(mask_y_end - mask_y_start, mask_x_end - mask_x_start),
                mode='bilinear', 
                align_corners=False
            )

            final_pseudo_image[:, :, mask_y_start:mask_y_end, mask_x_start:mask_x_end] = pseudo_image_resized

        
        pseudo_mask = final_pseudo_image * 255 - F.interpolate(
            (image * 255) / 3, size=(output_size, output_size), mode='bilinear', align_corners=False
        )

        
        final_mask = torch.where(
            pseudo_mask[:, 0, :, :] < pseudo_mask[:, 1, :, :],
            torch.zeros_like(pseudo_mask[:, 0, :, :]), 
            torch.ones_like(pseudo_mask[:, 0, :, :])  
        )

        #final_mask = final_mask.squeeze(0)

        return final_mask

    def on_validation_start(self):
        text_embedding = torch.cat(
            [
                self.text_embedding[:, 0:1],  
                self.embeddings_to_optimize[1].to(self.device),  
                self.text_embedding[:, 2:3],  
                self.embeddings_to_optimize[0].to(self.device),  
                self.text_embedding[:, 4:],  
            ],
            dim=1,
        )
        self.test_t_embedding = torch.cat([self.uncond_embedding, text_embedding])

    def on_validation_epoch_start(self):
        self.val_ious = []

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        # mask = mask[0]
        final_mask = self.get_patched_masks(
            image,
            self.config.test_mask_size,
        )

        ious = []
        for idx, part_name in enumerate(self.config.part_names):
            part_mask = torch.where(mask == idx, 1, 0).type(torch.uint8)
            if torch.all(part_mask == 0):
                continue
            iou = calculate_iou(
                torch.where(final_mask == idx, 1, 0).type(torch.uint8), part_mask
            )
            ious.append(iou)
            self.log(f"val {part_name} iou", iou.cpu(), on_step=True, sync_dist=True)
        mean_iou = sum(ious) / len(ious)
        self.val_ious.append(mean_iou)
        self.log("val mean iou", mean_iou.cpu(), on_step=True, sync_dist=True)
        return torch.tensor(0.0)

    def on_validation_epoch_end(self):
        epoch_mean_iou = sum(self.val_ious) / len(self.val_ious)
        if epoch_mean_iou >= self.max_val_iou:
            self.max_val_iou = epoch_mean_iou
            for i, embedding in enumerate(self.embeddings_to_optimize):
                torch.save(
                    embedding,
                    os.path.join(self.checkpoint_dir, f"embedding_{i}.pth"),
                )
        gc.collect()

    def on_test_start(self) -> None:
        self.stable_diffusion.setup(self.device)
        uncond_embedding, text_embedding = self.uncond_embedding.to(
            self.device
        ), self.text_embedding.to(self.device)
        embeddings_to_optimize = []
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.config.checkpoint_dir
        for i in range(self.num_parts):
            embedding = torch.load(
                os.path.join(self.checkpoint_dir, f"embedding_{i}.pth")
            )
            embeddings_to_optimize.append(embedding)
            
        print(len(embeddings_to_optimize))
        text_embedding = torch.cat(
            [
                text_embedding[:, 0:1],  
                embeddings_to_optimize[1].to(self.device),  
                text_embedding[:, 2:3],  
                embeddings_to_optimize[0].to(self.device),  
                text_embedding[:, 4:],  
            ],
            dim=1,
        )
        self.test_t_embedding = torch.cat([uncond_embedding, text_embedding])
        if self.config.save_test_predictions:
            self.distinct_colors = generate_distinct_colors(self.num_parts - 1)
            self.test_results_dir = os.path.join(
                self.config.output_dir,
                "test_results",
                self.logger.log_dir.split("/")[-1],
            )
            os.makedirs(self.test_results_dir)

    def test_step(self, batch, batch_idx):
        image, mask = batch
        mask_provided = not torch.all(mask == 0)
        # mask = mask[0]
        final_mask = self.get_patched_masks(
            image,
            self.config.test_mask_size,
        )


        if self.config.save_test_predictions:
            pseudo_image = self.stable_diffusion.inference_step(
                self.test_t_embedding,
                image,
                t=torch.tensor(self.config.test_t),
                generate_new_noise=True,
            )
            pseudo_image = pseudo_image.detach().cpu().permute(0, 2, 3, 1).numpy()
            pseudo_image = (pseudo_image * 255).round().astype("uint8")
            image1 = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            image1 = (image1 * 255).round().astype("uint8")
            image1 = image1[:, :, :, ::-1]
            
            
            for i in range(pseudo_image.shape[0]):
                
                img = pseudo_image[i]
                img2 = image1[i]
            
                
                img = Image.fromarray(img)
                img2 = Image.fromarray(img2)
            
                
                file_name = f"pseudo_{batch_idx * pseudo_image.shape[0] + i}.png"
                file_name2 = f"original_{batch_idx * image1.shape[0] + i}.png"
            
                
                file_path = os.path.join(self.test_results_dir, file_name)
                file_path2 = os.path.join(self.test_results_dir, file_name2)
            
                
                img.save(file_path)
                img2.save(file_path2)
                
                        
            final_mask = final_mask[0]

            final_mask = fill_holes_and_remove_small_areas(final_mask)

            final_mask = final_mask.to(mask.device)

            #eroded_final_mask, final_mask_boundary = get_boundry_and_eroded_mask(
            #    final_mask.cpu()
            #)
            
            open_final_mask = get_boundry_and_eroded_mask(
                final_mask.cpu()
            )
            
            final_mask.unsqueeze(0)
            
            colored_image = get_colored_segmentation(
                torch.tensor(open_final_mask),
                None,
                image[0].cpu(),
            )

            
            for i in range(image.shape[0]):
                Image.fromarray((255 * colored_image).type(torch.uint8).numpy()[:, :, ::-1]).save(
                    os.path.join(
                        self.test_results_dir, f"{batch_idx * image.shape[0] + i}.png"
                    )
                )
                Image.fromarray((255 * open_final_mask).astype(np.uint8)).save(
                    os.path.join(
                        self.test_results_dir, f"final_mask_{batch_idx * image.shape[0] + i}.png"
                    )
                )

        if mask_provided:
            for idx, part_name in enumerate(self.config.part_names):
                part_mask = torch.where(mask == idx, 1, 0).type(torch.uint8)
                if torch.all(part_mask == 0):
                    continue
                iou = calculate_iou(
                    torch.where(final_mask == idx, 1, 0).type(torch.uint8), part_mask
                )
                # self.ious[part_name].append(iou.cpu())
                self.log(
                    f"test {part_name} iou", iou.cpu(), on_step=True, sync_dist=True
                )

        return torch.tensor(0.0)

    def on_test_end(self) -> None:
        print("max val mean iou: ", self.max_val_iou)

    def configure_optimizers(self):
        parameters = [{"params": self.embeddings_to_optimize, "lr": self.config.lr}]
        optimizer = getattr(optim, self.config.optimizer)(
            parameters,
            lr=self.config.lr,
        )
        return optimizer



