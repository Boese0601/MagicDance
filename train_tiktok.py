"""
Train a control net
"""

import os
import sys
import argparse
import datetime
import numpy as np
import pdb
# torch
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torchvision import transforms as T
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
#from ema_pytorch import EMA
import kornia
from einops import rearrange
import einops

# distributed 
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP 

# data
from dataset import  tiktok_video_arnold_copy
from dataset.hdfs_io import hcopy, hexists
from dataset.transforms import RemoveWhite, CenterCrop

# utils
from utils.checkpoint import load_from_pretrain, save_checkpoint_ema
from utils.utils import set_seed, count_param, print_peak_memory, anal_tensor, merge_lists_by_index
from utils.lr_scheduler import LambdaLinearScheduler
from dataset.hdfs_io import hexists, hmkdir, hopen, hcopy
from langdetect import detect
# model
import imageio
from PIL import Image
from model_lib.ControlNet.cldm.model import create_model, instantiate_from_config
import copy
TORCH_VERSION = torch.__version__.split(".")[0]
FP16_DTYPE = torch.float16 if TORCH_VERSION == "1" else torch.bfloat16
print(f"TORCH_VERSION={TORCH_VERSION} FP16_DTYPE={FP16_DTYPE}")

# torch.cuda.OutOfMemoryError: CUDA out of memory. 
# Tried to allocate 120.00 MiB (GPU 0; 47.45 GiB total capacity; 43.81 GiB already allocated; 118.31 MiB free; 
# 44.16 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb 
# to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

def tensor_to_image(tensor):
    # Assuming tensor shape is [1, 3, 64, 64]
    # Convert the tensor to a numpy array and move the channel dimension to the last axis
    image_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Rescale the values from [-1, 1] to [0, 255]
    image_np = ((image_np + 1) * 0.5 * 255).astype('uint8')

    return Image.fromarray(image_np)

def build_mask_image(image,mask):
    binary_mask = torch.where(mask <= 0, 0.0, 1.0)
    result = binary_mask * image
    return (1.0 - binary_mask)[:,0,...].unsqueeze(1), result

def build_face_mask_image(image, face_mask):
    result = image * face_mask      
    return result, (1.0 - face_mask)


def is_english(text):
    try:
        lang = detect(text)
        return lang == 'en'
    except:
        return False
        
def write_list_to_file(lst, filename):
    with open(filename, 'w') as file:
        for item in lst:
            file.write(item + '\n')

def delete_zero_conv_in_controlmodel(state_dict):
    keys_to_delete = [key for key in state_dict.keys() if (key.startswith('control_model.zero_convs') or key.startswith('control_model.middle_block_out')) ]

    for key in keys_to_delete:
        del state_dict[key]
    return state_dict

# You can now use the modified state_dict without the deleted keys

def copy_diffusion_outputblocks(state_dict):

    new_state_dict = copy.deepcopy(state_dict)
    for key, value in state_dict.items():
        if key.startswith('model.diffusion_model.output_blocks'):
 
            new_key = key.replace('model.diffusion_model.output_blocks', 'control_model.output_blocks')
            new_state_dict[new_key] = value
    return new_state_dict

def load_state_dict(model, ckpt_path, reinit_hint_block=False, strict=True, map_location="cpu"):
    print(f"Loading model state dict from {ckpt_path} ...")
    state_dict = load_from_pretrain(ckpt_path, map_location=map_location)
    state_dict = state_dict.get('state_dict', state_dict)
    if reinit_hint_block:
        print("Ignoring hint block parameters from checkpoint!")
        for k in list(state_dict.keys()):
            if k.startswith("control_model.input_hint_block"):
                state_dict.pop(k)
    model.load_state_dict(state_dict, strict=strict)
    del state_dict   


def load_state_dict_pretrained_everything_mm(model, pretrained_everything_path, mm_ckpt_path,strict=False, map_location="cpu"):
    print(f"Loading pretrained_everything_path state dict from {pretrained_everything_path} ...")
    print(f"Loading motion module state dict from {mm_ckpt_path} ...")
    mm_state_dict = torch.load(mm_ckpt_path, map_location=map_location)
    mm_state_dict = mm_state_dict.get('state_dict', mm_state_dict)
    image_state_dict = load_from_pretrain(pretrained_everything_path, map_location=map_location)
    image_state_dict = image_state_dict.get('state_dict', image_state_dict)
    state_dict_final = merge_state_dict_mm(image_state_dict, mm_state_dict)
    model.load_state_dict(state_dict_final, strict=True)
    del image_state_dict, mm_state_dict, state_dict_final

def replace_keys_in_state_dict(state_dict,str1,str2):
    new_state_dict = {}
    for key, value in state_dict.items():
        if str1 in key:
            new_key = key.replace(str1, str2)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def merge_state_dict(image_state_dict,pose_state_dict_new):
    for key, value in pose_state_dict_new.items():
        if "pose_control_model" in key:
            image_state_dict[key] = value
        else:
            continue
    return image_state_dict

def merge_state_dict_mm(image_state_dict,mm_state_dict):
    for key, value in mm_state_dict.items():
        if "motion_modules" in key:
            if "down_blocks.0.motion_modules.0" in key:
                image_state_dict[key.replace("down_blocks.0.motion_modules.0","model.diffusion_model.input_blocks_motion_module.0.0")] = value
            elif "down_blocks.0.motion_modules.1" in key:
                image_state_dict[key.replace("down_blocks.0.motion_modules.1","model.diffusion_model.input_blocks_motion_module.1.0")] = value 
            elif "down_blocks.1.motion_modules.0" in key:
                image_state_dict[key.replace("down_blocks.1.motion_modules.0","model.diffusion_model.input_blocks_motion_module.2.0")] = value    
            elif "down_blocks.1.motion_modules.1" in key:
                image_state_dict[key.replace("down_blocks.1.motion_modules.1","model.diffusion_model.input_blocks_motion_module.3.0")] = value   
            elif "down_blocks.2.motion_modules.0" in key:
                image_state_dict[key.replace("down_blocks.2.motion_modules.0","model.diffusion_model.input_blocks_motion_module.4.0")] = value    
            elif "down_blocks.2.motion_modules.1" in key:
                image_state_dict[key.replace("down_blocks.2.motion_modules.1","model.diffusion_model.input_blocks_motion_module.5.0")] = value
            elif "down_blocks.3.motion_modules.0" in key:
                image_state_dict[key.replace("down_blocks.3.motion_modules.0","model.diffusion_model.input_blocks_motion_module.6.0")] = value    
            elif "down_blocks.3.motion_modules.1" in key:
                image_state_dict[key.replace("down_blocks.3.motion_modules.1","model.diffusion_model.input_blocks_motion_module.7.0")] = value         
        
            elif "up_blocks.0.motion_modules.0" in key:
                image_state_dict[key.replace("up_blocks.0.motion_modules.0","model.diffusion_model.output_blocks_motion_module.0.0")] = value    
            elif "up_blocks.0.motion_modules.1" in key:
                image_state_dict[key.replace("up_blocks.0.motion_modules.1","model.diffusion_model.output_blocks_motion_module.1.0")] = value 
            elif "up_blocks.0.motion_modules.2" in key:
                image_state_dict[key.replace("up_blocks.0.motion_modules.2","model.diffusion_model.output_blocks_motion_module.2.0")] = value 
            elif "up_blocks.1.motion_modules.0" in key:
                image_state_dict[key.replace("up_blocks.1.motion_modules.0","model.diffusion_model.output_blocks_motion_module.3.0")] = value    
            elif "up_blocks.1.motion_modules.1" in key:
                image_state_dict[key.replace("up_blocks.1.motion_modules.1","model.diffusion_model.output_blocks_motion_module.4.0")] = value 
            elif "up_blocks.1.motion_modules.2" in key:
                image_state_dict[key.replace("up_blocks.1.motion_modules.2","model.diffusion_model.output_blocks_motion_module.5.0")] = value 
            elif "up_blocks.2.motion_modules.0" in key:
                image_state_dict[key.replace("up_blocks.2.motion_modules.0","model.diffusion_model.output_blocks_motion_module.6.0")] = value    
            elif "up_blocks.2.motion_modules.1" in key:
                image_state_dict[key.replace("up_blocks.2.motion_modules.1","model.diffusion_model.output_blocks_motion_module.7.0")] = value 
            elif "up_blocks.2.motion_modules.2" in key:
                image_state_dict[key.replace("up_blocks.2.motion_modules.2","model.diffusion_model.output_blocks_motion_module.8.0")] = value 
            elif "up_blocks.3.motion_modules.0" in key:
                image_state_dict[key.replace("up_blocks.3.motion_modules.0","model.diffusion_model.output_blocks_motion_module.9.0")] = value    
            elif "up_blocks.3.motion_modules.1" in key:
                image_state_dict[key.replace("up_blocks.3.motion_modules.1","model.diffusion_model.output_blocks_motion_module.10.0")] = value 
            elif "up_blocks.3.motion_modules.2" in key:
                image_state_dict[key.replace("up_blocks.3.motion_modules.2","model.diffusion_model.output_blocks_motion_module.11.0")] = value 
        else:
            continue
    return image_state_dict

def load_state_dict_image_pose(model, image_ckpt_path, pose_ckpt_path, strict=False, map_location="cpu"):
    print(f"Loading appearance model state dict from {image_ckpt_path} ...")
    print(f"Loading pose model state dict from {pose_ckpt_path} ...")
    state_dict = load_from_pretrain(image_ckpt_path, map_location=map_location)
    state_dict = state_dict.get('state_dict', state_dict)
    image_state_dict = replace_keys_in_state_dict(state_dict,"control_model","appearance_control_model")
    # pdb.set_trace()
    pose_state_dict = load_from_pretrain(pose_ckpt_path, map_location=map_location)
    # pdb.set_trace()
    pose_state_dict = pose_state_dict.get('state_dict', pose_state_dict)
    # pdb.set_trace()
    pose_state_dict_new = replace_keys_in_state_dict(pose_state_dict,"control_model","pose_control_model")
    # pdb.set_trace()
    state_dict_final = merge_state_dict(image_state_dict,pose_state_dict_new)
    # pdb.set_trace()
    model.load_state_dict(state_dict_final, strict=strict)
    del state_dict, image_state_dict, pose_state_dict, pose_state_dict_new, state_dict_final

def load_state_dict_image_pose_mm(model, image_ckpt_path, pose_ckpt_path, mm_ckpt_path,strict=False, map_location="cpu"):
    print(f"Loading appearance model state dict from {image_ckpt_path} ...")
    print(f"Loading pose model state dict from {pose_ckpt_path} ...")
    print(f"Loading pose model state dict from {mm_ckpt_path} ...")
    pdb.set_trace()
    mm_state_dict = torch.load(mm_ckpt_path, map_location=map_location)
    # pdb.set_trace()
    mm_state_dict = mm_state_dict.get('state_dict', mm_state_dict)
    # pdb.set_trace()
    model.load_state_dict(mm_state_dict, strict=False)
    pdb.set_trace()

    state_dict = load_from_pretrain(image_ckpt_path, map_location=map_location)
    state_dict = state_dict.get('state_dict', state_dict)
    image_state_dict = replace_keys_in_state_dict(state_dict,"control_model","appearance_control_model")
    pose_state_dict = load_from_pretrain(pose_ckpt_path, map_location=map_location)
    pose_state_dict = pose_state_dict.get('state_dict', pose_state_dict)
    pose_state_dict_new = replace_keys_in_state_dict(pose_state_dict,"control_model","pose_control_model")


    state_dict_final = merge_state_dict(image_state_dict,pose_state_dict_new)
    model.load_state_dict(state_dict_final, strict=strict)
    del state_dict, image_state_dict, pose_state_dict, pose_state_dict_new, state_dict_final

def load_state_dict_reference_only(model, ckpt_path, reinit_hint_block=False, strict=True, map_location="cpu"):
    print(f"Loading model state dict from {ckpt_path} ...")
    state_dict = load_from_pretrain(ckpt_path, map_location=map_location)
    state_dict = state_dict.get('state_dict', state_dict)
    state_dict = delete_zero_conv_in_controlmodel(state_dict)
    new_state_dict = copy_diffusion_outputblocks(state_dict)
    if reinit_hint_block:
        print("Ignoring hint block parameters from checkpoint!")
        for k in list(state_dict.keys()):
            if k.startswith("control_model.input_hint_block"):
                state_dict.pop(k)
    
    model.load_state_dict(new_state_dict, strict=strict)
    del state_dict   

def load_state_dict_reference_only_mask(model, ckpt_path, reinit_hint_block=False, strict=True, map_location="cpu"):
    print(f"Loading model state dict from {ckpt_path} ...")
    state_dict = load_from_pretrain(ckpt_path, map_location=map_location)
    state_dict = state_dict.get('state_dict', state_dict)
    state_dict = delete_zero_conv_in_controlmodel(state_dict)
    new_state_dict = copy_diffusion_outputblocks(state_dict)
    conv_input_weight =  new_state_dict.pop('control_model.input_blocks.0.0.weight')

    
    if reinit_hint_block:
        print("Ignoring hint block parameters from checkpoint!")
        for k in list(state_dict.keys()):
            if k.startswith("control_model.input_hint_block"):
                state_dict.pop(k)
    
    model.load_state_dict(new_state_dict, strict=strict)
    # pdb.set_trace()
    model.control_model.input_blocks[0][0].weight.data[:,0,...] = torch.zeros(320,3,3)
    model.control_model.input_blocks[0][0].weight.data[:,1:,...] = conv_input_weight
    # pdb.set_trace()
    del state_dict   

def get_cond_inpaint(model, batch_data, device, batch_size=None, blank_mask=False):
    randommask = batch_data["randommask"][:batch_size]
    if blank_mask:
        randommask = torch.ones_like(randommask)
    masked_image = batch_data["image"][:batch_size] * (1-randommask)
    image = model.get_first_stage_encoding(model.encode_first_stage(masked_image.to(device)))
    mask = torch.nn.functional.interpolate(randommask.to(device), size=image.shape[-2:])
    inpaint = torch.cat((mask, image), 1)
    return inpaint, masked_image

def get_cond_control(args, batch_data, control_type, device, model=None, batch_size=None, train=True):
    control_type = copy.deepcopy(control_type)[0]
    if control_type == "body+hand+face" :
        if train:
            if args.train_dataset != "tiktok_video_mm":
                assert "pose_map" in batch_data
                pose_map = batch_data["pose_map"].cuda()
                c_cat = pose_map
                cond_image = batch_data["condition_image"].cuda()
                if args.mask_bg:
                    mask_bg, cond_image = build_mask_image(cond_image, batch_data['condition_mask'].cuda())
                if args.mask_face:
                    cond_image, mask_face = build_face_mask_image(cond_image, batch_data['face_mask'].cuda())
                if args.random_mask:
                    randommask = batch_data["randommask"].cuda()
                    cond_image = cond_image * (1-randommask)
                cond_image = model.get_first_stage_encoding(model.encode_first_stage(cond_image))
                if args.mask_bg:
                    mask = torch.nn.functional.interpolate(mask_bg.to(device), size=cond_image.shape[-2:])
                    cond_image = torch.cat((mask, cond_image), 1)
                if args.random_mask:
                    mask = torch.nn.functional.interpolate(randommask.to(device), size=cond_image.shape[-2:])
                    cond_image = torch.cat((mask, cond_image), 1)
                if args.mask_face:
                    mask = torch.nn.functional.interpolate(mask_face.to(device), size=cond_image.shape[-2:])
                    cond_image = torch.cat((mask, cond_image), 1)
                cond_img_cat = cond_image
            elif args.train_dataset == "tiktok_video_mm":
                assert "pose_map_list" in batch_data
                pose_map_list = batch_data["pose_map_list"].cuda()
                # pdb.set_trace()
                c_cat_list = [pose_map_list[:,i,:,:,:].cuda() for i in range(pose_map_list.shape[1])]
                # pdb.set_trace()
                cond_image = batch_data["condition_image"].cuda()

                cond_image = model.get_first_stage_encoding(model.encode_first_stage(cond_image))

                cond_img_cat = cond_image
            else:
                raise NotImplementedError
        else:
            if args.train_dataset != "tiktok_video_mm":
                assert "pose_map_list" in batch_data
                pose_map_list = batch_data["pose_map_list"]
                print("Get Inference Control Type")
                c_cat_list = [pose_map.cuda() for pose_map in pose_map_list]
                cond_image = batch_data["condition_image"].cuda()
                if args.mask_bg:
                    mask_bg, cond_image = build_mask_image(cond_image, batch_data['condition_mask'].cuda())
                if args.mask_face:
                    cond_image, mask_face = build_face_mask_image(cond_image, batch_data['face_mask'].cuda())
                if args.random_mask:
                    randommask = batch_data["randommask"].cuda()
                    cond_image = cond_image * (1-randommask)
                cond_image = model.get_first_stage_encoding(model.encode_first_stage(cond_image))
                if args.mask_bg:
                    mask = torch.nn.functional.interpolate(mask_bg.to(device), size=cond_image.shape[-2:])
                    cond_image = torch.cat((mask, cond_image), 1)
                if args.random_mask:
                    mask = torch.nn.functional.interpolate(randommask.to(device), size=cond_image.shape[-2:])
                    cond_image = torch.cat((mask, cond_image), 1)
                if args.mask_face:
                    mask = torch.nn.functional.interpolate(mask_face.to(device), size=cond_image.shape[-2:])
                    cond_image = torch.cat((mask, cond_image), 1)
                cond_img_cat = cond_image
            elif args.train_dataset == "tiktok_video_mm":
                assert "pose_map_list" in batch_data
                pose_map_list = batch_data["pose_map_list"].cuda()
                c_cat_list = [pose_map_list[:,i,:,:,:].cuda() for i in range(pose_map_list.shape[1])]
                cond_image = batch_data["condition_image"].cuda()
                cond_image = model.get_first_stage_encoding(model.encode_first_stage(cond_image))
                cond_img_cat = cond_image
            else:
                raise NotImplementedError
    else:
        raise NotImplementedError(f"cond_type={control_type} not supported!")

    if train:
        if args.train_dataset != "tiktok_video_mm":
            if args.control_dropout > 0:
                mask = torch.rand((c_cat.shape[0],1,1,1)) > args.control_dropout
                c_cat = c_cat * mask.type(torch.float32).to(device)
            return [ c_cat[:batch_size].to(device) ], [cond_img_cat[:batch_size].to(device) ]
        else:
            return_list = []
            for c_cat in c_cat_list:
                if args.control_dropout > 0:
                    mask = torch.rand((c_cat.shape[0],1,1,1)) > args.control_dropout
                    c_cat = c_cat * mask.type(torch.float32).to(device)
                # pdb.set_trace()
                return_list.append([c_cat[:batch_size].to(device)])
            # pdb.set_trace()
            return return_list, [cond_img_cat[:batch_size].to(device) ]
    else:
        return_list = []

        for c_cat in c_cat_list:
            if args.control_dropout > 0:
                mask = torch.rand((c_cat.shape[0],1,1,1)) > args.control_dropout
                c_cat = c_cat * mask.type(torch.float32).to(device)
            # pdb.set_trace()
            return_list.append([c_cat[:batch_size].to(device)])
        # pdb.set_trace()
        return return_list, [cond_img_cat[:batch_size].to(device) ]

def visualize(args, name, batch_data, tb_writer, infer_model, global_step, nSample, nTest=1):
    
    infer_model.eval()
    if args.train_dataset != "tiktok_video_mm":
        real_image_list = [real_image.cuda() for real_image in batch_data["image_list"]]
    else:
        image_list = batch_data["image_list"].to(args.device) # B,F,C,H,W

    rec_image_list = []
    if not args.v4:
        text = [" "] * nSample
    else:
        text = [" "] * nSample
    if not args.with_text:
        for i in range(len(text)):
            text[i] = ""
    if args.text_prompt is not None:
        for i in range(len(text)):
            text[i] = args.text_prompt

    if args.train_dataset == "tiktok_video_mm":
        c_cat_list, cond_img_cat = get_cond_control(args, batch_data, args.control_type, args.device, model=infer_model, batch_size=nSample,train=False)
        pose_map_list = [pose_map[0].cpu() for pose_map in c_cat_list]
        pose_tensor_list = torch.cat([c_cat[0] for c_cat in c_cat_list],0)
        c_cat_list = [pose_tensor_list]
        c_cross = infer_model.get_learned_conditioning(text)[:nSample]
        uc_cross = infer_model.get_unconditional_conditioning(nSample)
        c_cross = einops.repeat(c_cross, 'b h w -> (repeat b) h w', repeat=args.frame_num)
        uc_cross = einops.repeat(uc_cross, 'b h w -> (repeat b) h w', repeat=args.frame_num)
        cond_img_cat = [einops.repeat(cond_img_cat[0], 'b c h w -> (repeat b) c h w', repeat=args.frame_num)]
        c = {"c_concat": c_cat_list, "c_crossattn": [c_cross], "image_control": cond_img_cat}
        if args.control_mode == "controlnet_important":
            uc = {"c_concat": c_cat_list, "c_crossattn": [uc_cross]}
        else:
            uc = {"c_concat": c_cat_list, "c_crossattn": [uc_cross], "image_control":cond_img_cat}
        if args.wonoise:
            c['wonoise'] = True
            uc['wonoise'] = True
        else:
            c['wonoise'] = False
            uc['wonoise'] = False
        
        
        noise_shape = (c_cross.shape[0], infer_model.channels, infer_model.image_size, infer_model.image_size)
        noise = torch.randn(noise_shape, device=c_cross.device)
        with torch.cuda.amp.autocast(enabled=args.use_fp16, dtype=FP16_DTYPE):
            # for _ in range(nTest):
            infer_model.to(args.device)
            infer_model.eval()
            gene_img, _ = infer_model.sample_log(cond=c,
                                    batch_size=c_cross.shape[0], ddim=True,
                                    ddim_steps=50, eta=args.eta,
                                    unconditional_guidance_scale=7,
                                    unconditional_conditioning=uc,
                                    inpaint=None,
                                    x_T=noise
                                    )
            
            gene_img = infer_model.decode_first_stage( gene_img )
            
            gene_img = gene_img.clamp(-1, 1).cpu()
        image_list = rearrange(image_list, "b f c h w -> (b f) c h w")
        latent = infer_model.get_first_stage_encoding(infer_model.encode_first_stage(image_list))
        rec_image = infer_model.decode_first_stage(latent)

        gene_img_list = [gene_img[i,:,:,:].unsqueeze(0).cpu() for i in range(gene_img.shape[0])]
        rec_image_list = [rec_image[i,:,:,:].unsqueeze(0).cpu() for i in range(rec_image.shape[0])]

        grid = make_grid( torch.cat([rec_img.cpu() for rec_img in rec_image_list]+ pose_map_list + gene_img_list + [batch_data["condition_image"].cpu()]).clamp(-1,1).add(1).mul(0.5), nrow=len(rec_image_list))

        if args.with_text:
            write_list_to_file(text, f"{args.local_image_dir}/{name}_{global_step}.txt")
        save_image(grid, f"{args.local_image_dir}/{name}_{global_step}.jpg")


        imageio.mimsave(f"{args.local_image_dir}/{name}_{global_step}_gen.gif", [tensor_to_image(tensor.cpu()) for tensor in gene_img_list], duration=args.gif_time)  # Adjust the duration between frames as needed
        imageio.mimsave(f"{args.local_image_dir}/{name}_{global_step}_pose.gif", [tensor_to_image(tensor.clamp(-1, 1).cpu()) for tensor in pose_map_list], duration=args.gif_time)
        imageio.mimsave(f"{args.local_image_dir}/{name}_{global_step}_gt.gif", [tensor_to_image(tensor.clamp(-1, 1).cpu()) for tensor in rec_image_list], duration=args.gif_time)  # Adjust the duration between frames as needed

    else:
        c_cat_list, cond_img_cat = get_cond_control(args, batch_data, args.control_type, args.device, model=infer_model, batch_size=nSample,train=False)
        c_cross = infer_model.get_learned_conditioning(text)[:nSample]
        uc_cross = infer_model.get_unconditional_conditioning(nSample)
        gene_img_list = []
        pose_map_list = []
        for img_num in range(len(real_image_list)):
            print("Generate Image {} in {} images".format(img_num,len(real_image_list))) 
            # pdb.set_trace()
            c = {"c_concat": c_cat_list[img_num], "c_crossattn": [c_cross], "image_control":cond_img_cat}
            if args.control_mode == "controlnet_important":
                uc = {"c_concat": c_cat_list[img_num], "c_crossattn": [uc_cross]}
            else:
                uc = {"c_concat": c_cat_list[img_num], "c_crossattn": [uc_cross], "image_control":cond_img_cat}
            if args.wonoise:
                c['wonoise'] = True
                uc['wonoise'] = True
            else:
                c['wonoise'] = False
                uc['wonoise'] = False
            pose_map_list.append(c_cat_list[img_num][0][:,:3].cpu())
            if args.inpaint_unet:
                inpaint, masked_image = get_cond_inpaint(infer_model, batch_data, args.device, batch_size=nSample, blank_mask=False)
                inpaint_list = [masked_image.cpu()]
            else:
                inpaint = None
                inpaint_list = []
            noise_shape = (c_cross.shape[0], infer_model.channels, infer_model.image_size, infer_model.image_size)
            noise = torch.randn(noise_shape, device=c_cross.device)
            # generate images
            with torch.cuda.amp.autocast(enabled=args.use_fp16, dtype=FP16_DTYPE):
                # for _ in range(nTest):
                infer_model.to(args.device)
                infer_model.eval()
                gene_img, _ = infer_model.sample_log(cond=c,
                                        batch_size=nSample, ddim=True,
                                        ddim_steps=50, eta=args.eta,
                                        unconditional_guidance_scale=7,
                                        unconditional_conditioning=uc,
                                        inpaint=inpaint,
                                        x_T=noise,
                                        )
                gene_img = infer_model.decode_first_stage( gene_img )
                gene_img_list.append(gene_img.clamp(-1, 1).cpu())
            image = real_image_list[img_num][:nSample]
            # mask = real_mask_list[img_num][:nSample]

            latent = infer_model.get_first_stage_encoding(infer_model.encode_first_stage(image))
            rec_image = infer_model.decode_first_stage(latent)
            rec_image_list.append(rec_image)
        if args.random_mask:
            cond_image = batch_data["condition_image"].cpu() * (1-batch_data["randommask"].cpu())
        elif args.mask_bg:
            _, cond_image = build_mask_image(batch_data["condition_image"].cuda(), batch_data['condition_mask'].cuda())
        elif args.mask_face:
            cond_image, _ = build_face_mask_image(batch_data["condition_image"].cuda(), batch_data['face_mask'].cuda())
        else:
            cond_image = batch_data["condition_image"].cpu()
        grid = make_grid( torch.cat([rec_img.cpu() for rec_img in rec_image_list]+ pose_map_list + gene_img_list + [cond_image.cpu()]).clamp(-1,1).add(1).mul(0.5), nrow=len(rec_image_list))
        if args.with_text:
            write_list_to_file(text, f"{args.local_image_dir}/{name}_{args.control_type}_{global_step}.txt")
        save_image(grid, f"{args.local_image_dir}/{name}_{args.control_type}_{global_step}.jpg")
        tb_writer.add_image(f'{name}_{args.control_type}', grid, global_step)
        tb_writer.add_text(f'{name}_{args.control_type}_caption',f"{str(text)}", global_step )

def estimate_deviation(args, infer_model, tb_writer, global_step):
    def _calc_dist(model1, model2, keyword, replace=None):
        with torch.no_grad():
            model1 = model1.state_dict()
            model2 = model2.state_dict()
            if replace is None: replace = lambda x: x
            distance = 0
            keys = [k for k in model1.keys() if keyword in k and replace(k) in model2.keys()]
            for k in keys:
                distance += (model1[k].float() - model2[replace(k)].float()).pow(2).sum()
            return distance / len(keys)
    if args.local_rank == 0:
        log_output = "Deviation: "

def main(args):
    
    # ******************************
    # initialize training
    # ******************************
    # assign rank
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.rank = int(os.environ['RANK'])
    args.device = torch.device("cuda", args.local_rank)
    os.makedirs(args.local_image_dir,exist_ok=True)
    os.makedirs(args.local_log_dir,exist_ok=True)
    if args.rank == 0:
        print(args)

    # initial distribution comminucation
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # set seed for reproducibility
    set_seed(args.seed)

    # visdom / tensorboard
    if args.rank == 0:
        tb_writer = SummaryWriter(log_dir=args.local_log_dir)
    else:
        tb_writer = None
    
    # ******************************
    # create model
    # ******************************
    model = create_model(args.model_config).cpu()
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control
    model.to(args.local_rank)
    if args.local_rank == 0:
        print('Total base  parameters {:.02f}M'.format(count_param([model])))
    if args.ema_rate is not None and args.ema_rate > 0 and args.rank == 0:
        print(f"Creating EMA model at ema_rate={args.ema_rate}")
        model_ema = EMA(model, beta=args.ema_rate, update_after_step=0, update_every=1)
    else:
        model_ema = None

    # ******************************
    # load pre-trained models
    # ******************************
    optimizer_state_dict = None
    global_step = args.global_step
    args.resume_dir = args.resume_dir or args.output_dir
    if args.resume_dir is not None and hexists(os.path.join(args.resume_dir, "optimizer_state_latest.th")):
        print('find optimizer state dict from {} ...'.format(os.path.join(args.resume_dir, "optimizer_state_latest.th")))
        optimizer_state_dict = load_from_pretrain(os.path.join(args.resume_dir, "optimizer_state_latest.th"), map_location="cpu")
        if global_step == 0:
            global_step = optimizer_state_dict["step"]
        if args.local_rank == 0:
            assert hexists(os.path.join(args.resume_dir, f"model_state-{global_step}.th"))
            load_state_dict(model, os.path.join(args.resume_dir, f"model_state-{global_step}.th"))
        if args.rank == 0 and model_ema is not None:
            if hexists(os.path.join(args.resume_dir, f"model_state-{global_step}_ema_{args.ema_rate}.th")):
                load_state_dict(model_ema.ema_model, os.path.join(args.resume_dir, f"model_state-{global_step}_ema_{args.ema_rate}.th"))
            else:
                model_ema.copy_params_from_model_to_ema()
    else:
        # Initializing the model with SD weights
        if  args.image_pretrain_dir is None:
            assert args.init_path is not None
            if args.pose_pretrain_dir is not None:
                print("Load appearance coontrol model weights from init path.")
                print('find pose_pretrain model from {} ...'.format(os.path.join(args.pose_pretrain_dir, "optimizer_state_latest.th")))
                if not args.pose_pretrain_dir.endswith(".pth"):
                    optimizer_state_dict_pose = load_from_pretrain(os.path.join(args.pose_pretrain_dir, "optimizer_state_latest.th"), map_location="cpu")
                else:
                    optimizer_state_dict_pose = None
                if global_step == 0:
                    if not args.pose_pretrain_dir.endswith(".pth"):
                        global_step_pose = optimizer_state_dict_pose["step"]
                    else:
                        global_step_pose = None
                optimizer_state_dict_pose = None
                if args.local_rank == 0:
                    if not args.pose_pretrain_dir.endswith(".pth"):
                        assert hexists(os.path.join(args.pose_pretrain_dir, f"model_state-{global_step_pose}.th"))
                    if "model_lib/ControlNet/models/cldm_v15_reference_only_temporal_pose.yaml" in args.model_config:
                        print("Load pretrained reference only controlnet and pose controlnet for motion module+pose controlnet training") 
                        if args.pose_pretrain_dir.endswith(".pth"):
                            load_state_dict_image_pose(model, args.init_path, args.pose_pretrain_dir, strict=False)
                        else:
                            load_state_dict_image_pose(model, args.init_path, os.path.join(args.pose_pretrain_dir, f"model_state-{global_step_pose}.th"), strict=False)
                    else:
                        if args.pose_pretrain_dir.endswith(".pth"):
                            load_state_dict_image_pose(model, args.init_path, args.pose_pretrain_dir, strict=False)
                        else:
                            load_state_dict_image_pose(model, args.init_path, os.path.join(args.pose_pretrain_dir, f"model_state-{global_step_pose}.th"), strict=False)
                if args.rank == 0 and model_ema is not None:
                    model_ema.copy_params_from_model_to_ema()
            else:
                if args.pretrained_everything is not None:
                    print('find model state dict from {} ...'.format(args.pretrained_everything))
                    if args.local_rank == 0:
                        assert hexists(args.pretrained_everything)
                        if args.mm_pretrain_dir is not None:
                            print('find pertrained checkpoint from {} ...'.format(args.pretrained_everything))
                            print('find motion module from {} ...'.format(args.mm_pretrain_dir))
                            load_state_dict_pretrained_everything_mm(model, args.pretrained_everything, args.mm_pretrain_dir, strict=False)
                        else:
                            load_state_dict(model, args.pretrained_everything ,strict=False)
                    if model_ema is not None:
                        model_ema.copy_params_from_model_to_ema()
                    

                else:
                    print("Load model weights from init path.")
                    if args.model_config == "model_lib/ControlNet/models/cldm_v15_reference_only.yaml" or  args.model_config == "./model_lib/ControlNet/models/cldm_v15_reference_only.yaml":
                        print("Reference Only Control.")
                        load_state_dict_reference_only(model, args.init_path, reinit_hint_block=args.reinit_hint_block, strict=True)
                    elif args.model_config == "model_lib/ControlNet/models/cldm_v15_reference_only_mask.yaml" or  args.model_config == "./model_lib/ControlNet/models/cldm_v15_reference_only_mask.yaml":
                        print("Reference Only Control with Mask.")
                        load_state_dict_reference_only_mask(model, args.init_path, reinit_hint_block=args.reinit_hint_block, strict=False)
                    else:
                        load_state_dict(model, args.init_path, reinit_hint_block=args.reinit_hint_block, strict=False)
                    if model_ema is not None:
                        model_ema.copy_params_from_model_to_ema()
        # Load weights from Image Model in Step 1
        else:
            if args.pose_pretrain_dir is None:
                print('find image_pretrain model from {} ...'.format(os.path.join(args.image_pretrain_dir, "optimizer_state_latest.th")))
                optimizer_state_dict = load_from_pretrain(os.path.join(args.image_pretrain_dir, "optimizer_state_latest.th"), map_location="cpu")
                if global_step == 0:
                    global_step = optimizer_state_dict["step"]
                optimizer_state_dict = None
                if args.local_rank == 0:
                    assert hexists(os.path.join(args.image_pretrain_dir, f"model_state-{global_step}.th"))
                    if "model_lib/ControlNet/models/cldm_v15_reference_only_temporal.yaml" in args.model_config:
                        print("Load pretrained reference only controlnet for motion module training") 
                        load_state_dict(model, os.path.join(args.image_pretrain_dir, f"model_state-{global_step}.th"),strict=False)
                    else:
                        load_state_dict(model, os.path.join(args.image_pretrain_dir, f"model_state-{global_step}.th"),strict=False)
                if args.rank == 0 and model_ema is not None:
                    if hexists(os.path.join(args.image_pretrain_dir, f"model_state-{global_step}_ema_{args.ema_rate}.th")):
                        load_state_dict(model_ema.ema_model, os.path.join(args.image_pretrain_dir, f"model_state-{global_step}_ema_{args.ema_rate}.th"),strict=False)
                    else:
                        model_ema.copy_params_from_model_to_ema()
            else:
                if args.mm_pretrain_dir is not None:
                    print('find image_pretrain model from {} ...'.format(os.path.join(args.image_pretrain_dir, "optimizer_state_latest.th")))
                    print('find pose_pretrain model from {} ...'.format(os.path.join(args.pose_pretrain_dir, "optimizer_state_latest.th")))
                    print('find motion module from {} ...'.format(args.mm_pretrain_dir))
                    optimizer_state_dict_image = load_from_pretrain(os.path.join(args.image_pretrain_dir, "optimizer_state_latest.th"), map_location="cpu")
                    if not args.pose_pretrain_dir.endswith(".pth"):
                        optimizer_state_dict_pose = load_from_pretrain(os.path.join(args.pose_pretrain_dir, "optimizer_state_latest.th"), map_location="cpu")
                    else:
                        optimizer_state_dict_pose = None
                    if global_step == 0:
                        global_step_image = optimizer_state_dict_image["step"]
                        if not args.pose_pretrain_dir.endswith(".pth"):
                            global_step_pose = optimizer_state_dict_pose["step"]
                        else:
                            global_step_pose = None
                    optimizer_state_dict_image, optimizer_state_dict_pose = None, None
                    if args.local_rank == 0:
                        assert hexists(os.path.join(args.image_pretrain_dir, f"model_state-{global_step_image}.th"))
                        if not args.pose_pretrain_dir.endswith(".pth"):
                            assert hexists(os.path.join(args.pose_pretrain_dir, f"model_state-{global_step_pose}.th"))
                        if "model_lib/ControlNet/models/cldm_v15_reference_only_temporal_pose.yaml" in args.model_config:
                            print("Load pretrained reference only controlnet, pose controlnet and motion module for motion module+pose controlnet training") 
                            if args.pose_pretrain_dir.endswith(".pth"):
                                load_state_dict_image_pose_mm(model, os.path.join(args.image_pretrain_dir, f"model_state-{global_step_image}.th"), args.pose_pretrain_dir, args.mm_pretrain_dir, strict=False)
                            else:
                                load_state_dict_image_pose_mm(model, os.path.join(args.image_pretrain_dir, f"model_state-{global_step_image}.th"), os.path.join(args.pose_pretrain_dir, f"model_state-{global_step_pose}.th"), args.mm_pretrain_dir, strict=False)
                        else:
                            if args.pose_pretrain_dir.endswith(".pth"):
                                load_state_dict_image_pose_mm(model, os.path.join(args.image_pretrain_dir, f"model_state-{global_step_image}.th"), args.pose_pretrain_dir, args.mm_pretrain_dir, strict=False)
                            else:
                                load_state_dict_image_pose_mm(model, os.path.join(args.image_pretrain_dir, f"model_state-{global_step_image}.th"), os.path.join(args.pose_pretrain_dir, f"model_state-{global_step_pose}.th"), args.mm_pretrain_dir, strict=False)
                    if args.rank == 0 and model_ema is not None:
                        model_ema.copy_params_from_model_to_ema()
                else:
                    print('find image_pretrain model from {} ...'.format(os.path.join(args.image_pretrain_dir, "optimizer_state_latest.th")))
                    print('find pose_pretrain model from {} ...'.format(os.path.join(args.pose_pretrain_dir, "optimizer_state_latest.th")))
                    optimizer_state_dict_image = load_from_pretrain(os.path.join(args.image_pretrain_dir, "optimizer_state_latest.th"), map_location="cpu")
                    if not args.pose_pretrain_dir.endswith(".pth"):
                        optimizer_state_dict_pose = load_from_pretrain(os.path.join(args.pose_pretrain_dir, "optimizer_state_latest.th"), map_location="cpu")
                    else:
                        optimizer_state_dict_pose = None
                    if global_step == 0:
                        global_step_image = optimizer_state_dict_image["step"]
                        if not args.pose_pretrain_dir.endswith(".pth"):
                            global_step_pose = optimizer_state_dict_pose["step"]
                        else:
                            global_step_pose = None
                    optimizer_state_dict_image, optimizer_state_dict_pose = None, None
                    if args.local_rank == 0:
                        assert hexists(os.path.join(args.image_pretrain_dir, f"model_state-{global_step_image}.th"))
                        if not args.pose_pretrain_dir.endswith(".pth"):
                            assert hexists(os.path.join(args.pose_pretrain_dir, f"model_state-{global_step_pose}.th"))
                        if "model_lib/ControlNet/models/cldm_v15_reference_only_temporal_pose.yaml" in args.model_config:
                            print("Load pretrained reference only controlnet and pose controlnet for motion module+pose controlnet training") 
                            if args.pose_pretrain_dir.endswith(".pth"):
                                load_state_dict_image_pose(model, os.path.join(args.image_pretrain_dir, f"model_state-{global_step_image}.th"), args.pose_pretrain_dir, strict=False)
                            else:
                                load_state_dict_image_pose(model, os.path.join(args.image_pretrain_dir, f"model_state-{global_step_image}.th"), os.path.join(args.pose_pretrain_dir, f"model_state-{global_step_pose}.th"), strict=False)
                        else:
                            if args.pose_pretrain_dir.endswith(".pth"):
                                load_state_dict_image_pose(model, os.path.join(args.image_pretrain_dir, f"model_state-{global_step_image}.th"), args.pose_pretrain_dir, strict=False)
                            else:
                                load_state_dict_image_pose(model, os.path.join(args.image_pretrain_dir, f"model_state-{global_step_image}.th"), os.path.join(args.pose_pretrain_dir, f"model_state-{global_step_pose}.th"), strict=False)
                    if args.rank == 0 and model_ema is not None:
                        model_ema.copy_params_from_model_to_ema()
        global_step = 0
        torch.cuda.empty_cache()


    # ******************************
    # create optimizer
    # ******************************
    if args.finetune_all:
        params = list(model.control_model.parameters())
        params += list(model.model.diffusion_model.parameters())
        try:
            params += list(model.image_control_model.parameters())
        except:
            pass
        print("Finetune All.")
    elif args.finetune_imagecond_unet:
        params = list(model.model.diffusion_model.parameters())
        try:
            params += list(model.image_control_model.parameters())
        except: 
            pass
        for p in model.control_model.parameters():
            p.requires_grad_(False)
        print("Finetune Unet and image controlnet, freeze pose controlnet")
    elif args.finetune_attn:
        for b in list(model.model.diffusion_model.input_blocks) + [model.model.diffusion_model.middle_block] + list(model.model.diffusion_model.output_blocks) + list(model.model.diffusion_model.out):
            for p in b.parameters():
                p.requires_grad_(False)
        self_attn_parameters = []
        for name, param in model.model.diffusion_model.named_parameters():
            if "attn1" in name:
                param.requires_grad_(True)
                self_attn_parameters.append(param)
        if "cldm_v15_reference_only_pose.yaml" in args.model_config:
            params = list(model.appearance_control_model.parameters())
            params += list(model.pose_control_model.parameters())
            print("Train controlnet and attention layers in UNet")
        else:
            params = list(model.control_model.parameters())
        params+= self_attn_parameters
        if not args.sd_locked:
            params += list(model.model.diffusion_model.output_blocks.parameters())
            params += list(model.model.diffusion_model.out.parameters())
    elif args.finetune_control:
        for b in list(model.model.diffusion_model.input_blocks) + [model.model.diffusion_model.middle_block] + list(model.model.diffusion_model.output_blocks) + list(model.model.diffusion_model.out):
            for p in b.parameters():
                p.requires_grad_(False)
        if ("cldm_v15_reference_only_pose.yaml" in args.model_config) or ("cldm_v15_reference_only_pose_mask.yaml") in args.model_config:
            params = list(model.appearance_control_model.parameters())
            if not args.fix_hint:
                params += list(model.pose_control_model.parameters())
                print("Train controlnet layers in UNet")
            else:
                # TODO
                for b in list(model.pose_control_model.input_hint_block):
                    for p in b.parameters():
                        p.requires_grad_(False)
                # params += list(model.pose_control_model.parameters())
                params += list(model.pose_control_model.input_blocks.parameters())
                params += list(model.pose_control_model.middle_block.parameters())
                params += list(model.pose_control_model.middle_block_out.parameters())  
                params += list(model.pose_control_model.zero_convs.parameters())
                print("Train controlnet layers in UNet, fix hint block in pose controlnet.")
        else:
            params = list(model.control_model.parameters())
        if not args.sd_locked:
            params += list(model.model.diffusion_model.output_blocks.parameters())
            params += list(model.model.diffusion_model.out.parameters())
    elif args.finetune_pose_only:
        for b in list(model.model.diffusion_model.input_blocks) + [model.model.diffusion_model.middle_block] + list(model.model.diffusion_model.output_blocks) + list(model.model.diffusion_model.out) + list(model.appearance_control_model.input_blocks) + [model.appearance_control_model.middle_block] + list(model.appearance_control_model.output_blocks) + list(model.appearance_control_model.input_hint_block):
            for p in b.parameters():
                p.requires_grad_(False)
        if ("cldm_v15_reference_only_pose.yaml" in args.model_config) or ("cldm_v15_reference_only_pose_mask.yaml") in args.model_config:
            params = list(model.pose_control_model.parameters())
            print("Train pose controlnet layers in UNet, freeze appearance")
        else:
            params = list(model.control_model.parameters())
        if not args.sd_locked:
            params += list(model.model.diffusion_model.output_blocks.parameters())
            params += list(model.model.diffusion_model.out.parameters())
    elif args.finetune_reference_only:
        for b in list(model.model.diffusion_model.input_blocks) + [model.model.diffusion_model.middle_block] + list(model.model.diffusion_model.output_blocks) + list(model.model.diffusion_model.out) + list(model.pose_control_model.input_blocks) + [model.pose_control_model.middle_block] + list(model.pose_control_model.input_hint_block) + list(model.pose_control_model.middle_block_out)  + list(model.pose_control_model.zero_convs):
            for p in b.parameters():
                p.requires_grad_(False)
        if ("cldm_v15_reference_only_pose.yaml" in args.model_config) or ("cldm_v15_reference_only_pose_mask.yaml") in args.model_config:
            params = list(model.appearance_control_model.parameters())
            print("Train appearance controlnet layers in UNet, freeze pose controlnet")
        else:
            params = list(model.control_model.parameters())
        if not args.sd_locked:
            params += list(model.model.diffusion_model.output_blocks.parameters())
            params += list(model.model.diffusion_model.out.parameters())
    elif args.finetune_mm:
        if args.image_finetune: # As described in Animatediff, train motion module then finetune image Unet layers.
            for b in list(model.model.diffusion_model.input_blocks) + [model.model.diffusion_model.middle_block] + list(model.model.diffusion_model.output_blocks) + list(model.model.diffusion_model.out):
                for p in b.parameters():
                    p.requires_grad_(False)
            print("Freeze SD UNet")
            if ("cldm_v15_reference_only_temporal_pose.yaml") in args.model_config:
                for b in list(model.appearance_control_model.input_blocks) + [model.appearance_control_model.middle_block] + list(model.appearance_control_model.output_blocks):
                    for p in b.parameters():
                        p.requires_grad_(True)
                print("Train appearance controlnet")
                for b in list(model.pose_control_model.input_blocks) + [model.pose_control_model.middle_block] + list(model.pose_control_model.input_hint_block) + list(model.pose_control_model.middle_block_out)  + list(model.pose_control_model.zero_convs):
                    for p in b.parameters():
                        p.requires_grad_(True)
                print("Train pose controlnet")
                for b in list(model.model.diffusion_model.input_blocks_motion_module) + list(model.model.diffusion_model.output_blocks_motion_module):
                    for p in b.parameters():
                        p.requires_grad_(False)
                print("Freeze Motion Module")
                params = list(model.pose_control_model.parameters())
                params += list(model.appearance_control_model.input_blocks.parameters())
                params += list(model.appearance_control_model.middle_block.parameters())
                params += list(model.appearance_control_model.output_blocks.parameters())
        elif args.image_finetune_unet:
            for b in list(model.model.diffusion_model.input_blocks) + [model.model.diffusion_model.middle_block] + list(model.model.diffusion_model.output_blocks) + list(model.model.diffusion_model.out):
                for p in b.parameters():
                    p.requires_grad_(True)
            print("Train SD UNet")
            if ("cldm_v15_reference_only_temporal_pose.yaml") in args.model_config:
                for b in list(model.appearance_control_model.input_blocks) + [model.appearance_control_model.middle_block] + list(model.appearance_control_model.output_blocks):
                    for p in b.parameters():
                        p.requires_grad_(False)
                print("Freeze appearance controlnet")
                for b in list(model.pose_control_model.input_blocks) + [model.pose_control_model.middle_block] + list(model.pose_control_model.input_hint_block) + list(model.pose_control_model.middle_block_out)  + list(model.pose_control_model.zero_convs):
                    for p in b.parameters():
                        p.requires_grad_(False)
                print("Freeze pose controlnet")
                for b in list(model.model.diffusion_model.input_blocks_motion_module) + list(model.model.diffusion_model.output_blocks_motion_module):
                    for p in b.parameters():
                        p.requires_grad_(False)
                print("Freeze Motion Module")
                params = list(model.model.diffusion_model.parameters())

        elif args.unet_pose_app:
            for b in list(model.model.diffusion_model.input_blocks) + [model.model.diffusion_model.middle_block] + list(model.model.diffusion_model.output_blocks) + list(model.model.diffusion_model.out):
                for p in b.parameters():
                    p.requires_grad_(True)
            print("Train SD UNet")
            if ("cldm_v15_reference_only_temporal_pose.yaml") in args.model_config:
                for b in list(model.appearance_control_model.input_blocks) + [model.appearance_control_model.middle_block] + list(model.appearance_control_model.output_blocks):
                    for p in b.parameters():
                        p.requires_grad_(True)
                print("Train appearance controlnet")
                for b in list(model.pose_control_model.input_blocks) + [model.pose_control_model.middle_block] + list(model.pose_control_model.input_hint_block) + list(model.pose_control_model.middle_block_out)  + list(model.pose_control_model.zero_convs):
                    for p in b.parameters():
                        p.requires_grad_(True)
                print("Train pose controlnet")
                for b in list(model.model.diffusion_model.input_blocks_motion_module) + list(model.model.diffusion_model.output_blocks_motion_module):
                    for p in b.parameters():
                        p.requires_grad_(False)
                print("Freeze Motion Module")
                params = list(model.model.diffusion_model.parameters())
                params += list(model.pose_control_model.parameters())
                params += list(model.appearance_control_model.input_blocks.parameters())
                params += list(model.appearance_control_model.middle_block.parameters())
                params += list(model.appearance_control_model.output_blocks.parameters())

        else:
            for b in list(model.model.diffusion_model.input_blocks) + [model.model.diffusion_model.middle_block] + list(model.model.diffusion_model.output_blocks) + list(model.model.diffusion_model.out):
                for p in b.parameters():
                    p.requires_grad_(False)
            print("Freeze SD UNet")
            if ("cldm_v15_reference_only_temporal.yaml" in args.model_config):
                for b in list(model.control_model.input_blocks) + [model.control_model.middle_block] + list(model.control_model.output_blocks) + list(model.control_model.input_hint_block):
                    for p in b.parameters():
                        p.requires_grad_(False)
                print("Freeze appearance controlnet")
                params = list(model.model.diffusion_model.input_blocks_motion_module.parameters())  + list(model.model.diffusion_model.output_blocks_motion_module.parameters())
                print("Train motion module in UNet")

            if ("cldm_v15_reference_only_temporal_pose.yaml") in args.model_config:
                if not args.finetune_reference:
                    for b in list(model.appearance_control_model.input_blocks) + [model.appearance_control_model.middle_block] + list(model.appearance_control_model.output_blocks) + list(model.appearance_control_model.input_hint_block):
                        for p in b.parameters():
                            p.requires_grad_(False)
                    print("Freeze appearance controlnet")
                else:
                    for b in list(model.appearance_control_model.input_blocks) + [model.appearance_control_model.middle_block] + list(model.appearance_control_model.output_blocks):
                        for p in b.parameters():
                            p.requires_grad_(True)
                    print("Train appearance controlnet")
                if args.finetune_mm_only:
                    for b in list(model.pose_control_model.input_blocks) + [model.pose_control_model.middle_block] + list(model.pose_control_model.input_hint_block) + list(model.pose_control_model.middle_block_out)  + list(model.pose_control_model.zero_convs):
                        for p in b.parameters():
                            p.requires_grad_(False)
                    print("Freeze pose controlnet")

                params = list(model.model.diffusion_model.input_blocks_motion_module.parameters()) + list(model.model.diffusion_model.output_blocks_motion_module.parameters())
                print("Train Motion Module")
                if not args.finetune_mm_only:
                    params += list(model.pose_control_model.parameters())
                    print("Train Pose controlnet")
                if args.finetune_reference:
                    params += list(model.appearance_control_model.input_blocks.parameters())
                    params += list(model.appearance_control_model.middle_block.parameters())
                    params += list(model.appearance_control_model.output_blocks.parameters())
                    print("Train reference controlnet in UNet")
        if not args.sd_locked:
            params += list(model.model.diffusion_model.output_blocks.parameters())
            params += list(model.model.diffusion_model.out.parameters())
    else:
        for b in list(model.model.diffusion_model.input_blocks) + [model.model.diffusion_model.middle_block] + list(model.model.diffusion_model.output_blocks) + list(model.model.diffusion_model.out):
            for p in b.parameters():
                p.requires_grad_(False)
        params = list(model.control_model.parameters())
        try:
            params += list(model.image_control_model.parameters())
        except:
            pass
        if not args.sd_locked:
            params += list(model.model.diffusion_model.output_blocks.parameters())
            params += list(model.model.diffusion_model.out.parameters())
        print("Train controlnet")

    optimizer = ZeroRedundancyOptimizer(
        params = params,
        lr=args.lr,
        optimizer_class=torch.optim.AdamW,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.use_fp16 and FP16_DTYPE==torch.float16))
    
    # load optimizer state dict
    if optimizer_state_dict is not None and args.load_optimizer_state:
        try:
            optimizer.load_state_dict(optimizer_state_dict["state_dict"])
            if 'scaler_state_dict' in optimizer_state_dict: 
                scaler.load_state_dict(optimizer_state_dict["scaler_state_dict"])
            print("load optimizer done")
        except:
            print("WARNING: load optimizer failed")
        del optimizer_state_dict

    # create lr scheduler
    lr_scheduler = LambdaLinearScheduler(warm_up_steps=[ args.lr_anneal_steps ], cycle_lengths=[ 10000000000000 ], # incredibly large number to prevent corner cases
                                            f_start=[ 1.e-6 ], f_max=[ 1. ], f_min=[ 1. ])
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scheduler.schedule)
    lr_scheduler.step(global_step)

    # ******************************
    # create DDP model
    # ******************************
    if args.compile and TORCH_VERSION == "2":
        model = torch.compile(model)
    torch.cuda.set_device(args.local_rank)
    model = DDP(model, 
                device_ids=[args.local_rank], 
                output_device=args.local_rank,
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=True,
                gradient_as_bucket_view=True # will save memory
    )
    print_peak_memory("Max memory allocated after creating DDP", args.local_rank)

    # ******************************
    # create dataset and dataloader
    # ******************************
    if not args.v4:
        train_image_transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.RandomResizedCrop(
                args.image_size*8,
                scale=(0.9, 1.0), ratio=(1., 1.),
                interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(), 
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
        test_image_transform =  T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.RandomResizedCrop(
                args.image_size*8,
                scale=(1.0, 1.0), ratio=(1., 1.),
                interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(), 
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
        train_pose_transform = T.Compose([
        T.RandomResizedCrop(
                args.image_size*8,
                scale=(0.9, 1.0), ratio=(1., 1.),
                interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(), 
    ])
        test_pose_transform =  T.Compose([
        T.RandomResizedCrop(
                args.image_size*8,
                scale=(1.0, 1.0), ratio=(1., 1.),
                interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(), 
    ])
        # train_image_transform = T.Compose([
        #     T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        #     CenterCrop(),
        #     T.Resize((args.image_size*8, args.image_size*8)),
        #     T.ToTensor(), 
        #     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])
        # test_image_transform = T.Compose([
        #     T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        #     CenterCrop(),
        #     T.Resize((args.image_size*8, args.image_size*8)),
        #     T.ToTensor(), 
        #     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])
        # train_pose_transform = None
        # test_pose_transform = None
    else:
        train_image_transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.RandomResizedCrop(
                args.image_size*8,
                scale=(0.9, 1.0), ratio=(1., 1.),
                interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(), 
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
        test_image_transform =  T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.RandomResizedCrop(
                args.image_size*8,
                scale=(1.0, 1.0), ratio=(1., 1.),
                interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(), 
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
        train_pose_transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.RandomResizedCrop(
                args.image_size*8,
                scale=(0.9, 1.0), ratio=(1., 1.),
                interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(), 
    ])
        test_pose_transform =  T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.RandomResizedCrop(
                args.image_size*8,
                scale=(1.0, 1.0), ratio=(1., 1.),
                interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(), 
    ])
    if args.train_dataset == "tiktok_video_arnold":
        dataset_cls = getattr(tiktok_video_arnold_copy, args.train_dataset)
        test_dataset_cls = getattr(tiktok_video_arnold_copy, args.train_dataset+'_val')
        image_dataset = dataset_cls(transform=train_image_transform,
                                rank=args.rank,
                                world_size=args.world_size,
                                control_type=args.control_type,
                                inpaint=args.inpaint_unet,
                                train=True,
                                img_bin_limit=args.img_bin_limit,
                                random_mask=args.random_mask,
                                v4=args.v4,
                                pose_transform=train_pose_transform,
                                )
        test_image_dataset = test_dataset_cls(transform=test_image_transform,
                                rank=args.rank,
                                world_size=args.world_size,
                                control_type=args.control_type,
                                inpaint=args.inpaint_unet,
                                train=False,
                                img_bin_limit=args.img_bin_limit,
                                random_mask=args.random_mask,
                                v4=args.v4,
                                pose_transform=test_pose_transform,
                                )
    else:
        raise NotImplementedError
    

    image_dataloader = DataLoader(image_dataset, 
                                  batch_size=args.train_batch_size,
                                  num_workers=args.num_workers,
                                  drop_last=True,
                                  pin_memory=True)
    image_dataloader_iter = iter(image_dataloader)


    test_image_dataloader = DataLoader(test_image_dataset, 
                                  batch_size=args.val_batch_size,
                                  num_workers=1,
                                  pin_memory=True)
    test_image_dataloader_iter = iter(test_image_dataloader)
    if args.local_rank == 0:
        print(f"image dataloader created: dataset={args.train_dataset} batch_size={args.train_batch_size} control_type={args.control_type}")



    dist.barrier()
    local_step = 0
    loss_list = []
    first_print = True
    infer_model = model.module if hasattr(model, "module") else model
    print(f"[rank{args.rank}] start training loop!")
    estimate_deviation(args, infer_model, tb_writer, global_step)
    print("save step:", args.save_steps)
    
    for itr in range(global_step, args.num_train_steps):

        # Get input
        #batch_data = next(image_dataloader_iter)
        try: 
            batch_data=next(image_dataloader_iter)
        except Exception as e:
            image_dataloader_iter = iter(image_dataloader)
            batch_data=next(image_dataloader_iter)
        #print("Batch size:",len(batch_data),", num images:",len(batch_data["image"]),", num poses:",len(batch_data["pose_map"]))
        
        # pdb.set_trace()
        # print("pose_map",batch_data["pose_map"].shape)
        with torch.no_grad():
            if args.train_dataset != "tiktok_video_mm":
                #print("tiktok_video_arnold")
                image = batch_data["image"].to(args.device) #256, -1~1
            else:
                image_list = batch_data["image_list"].to(args.device) # B,F,C,H,W

            if not args.v4:
                text = [" "] * args.train_batch_size
            else:
                text = [" "] * args.train_batch_size
            for i in range(len(text)):
                if not args.with_text:
                    text[i] = ""
                else:
                    if np.random.random() <= args.empty_text_prob:
                        text[i] = ""
            
            if args.train_dataset != "tiktok_video_mm":
                x = infer_model.get_first_stage_encoding(infer_model.encode_first_stage(image))
            else:
                image_list = rearrange(image_list, "b f c h w -> (b f) c h w")
                x = infer_model.get_first_stage_encoding(infer_model.encode_first_stage(image_list))
            c_cat_list,cond_img_cat = get_cond_control(args, batch_data, args.control_type, args.device, model=infer_model, train=True)
            
            if args.train_dataset == "tiktok_video_mm":
                pose_tensor_list = torch.cat([c_cat[0] for c_cat in c_cat_list],0)
                c_cat_list = [pose_tensor_list]
            
            c_cross = infer_model.get_learned_conditioning(text)
            if args.train_dataset == "tiktok_video_mm":
                c_cross = einops.repeat(c_cross, 'b h w -> (repeat b) h w', repeat=args.frame_num)
                cond_img_cat = [einops.repeat(cond_img_cat[0], 'b c h w -> (repeat b) c h w', repeat=args.frame_num)]
            cond = {"c_concat": c_cat_list, "c_crossattn": [c_cross], "image_control": cond_img_cat}
            if args.wonoise:
                cond['wonoise'] = True
            else:
                cond['wonoise'] = False
            if args.inpaint_unet:
                blank_mask = np.random.random() <= args.blank_mask_prob
                cond["inpaint"] = get_cond_inpaint(infer_model, batch_data, args.device, blank_mask=blank_mask)[0] # will be concat to x for inpainting

        # Foward
        model.train()
        with torch.cuda.amp.autocast(enabled=args.use_fp16, dtype=FP16_DTYPE):
            # pdb.set_trace()
            loss, loss_dict = model(x, cond)
            loss_list.append(loss.item())
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
        if first_print:
            print_peak_memory("Max memory allocated after FOWARD first step", args.local_rank)
        
        # Backward

        scaler.scale(loss).backward()
 
        local_step += 1
        if first_print:
            print_peak_memory("Max memory allocated after BACKWARD first step", args.local_rank)
        
        # Optimize
        if local_step % args.gradient_accumulation_steps == 0:
            if not scaler.is_enabled():
                # pdb.set_trace()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            # pdb.set_trace()

            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)   
            if model_ema is not None:
                model_ema.update()
            local_step = 0
            global_step += 1

        # log losses
        if args.local_rank == 0 and args.logging_steps > 0 and (global_step % args.logging_steps == 0 or global_step < 10):
            lr = optimizer.param_groups[0]['lr']
            logging_loss = np.mean(loss_list) if len(loss_list) > 0 else 0
            memory_usage = torch.cuda.max_memory_allocated(args.local_rank) // 1e6
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + 
                f" Step {global_step:06}: loss {logging_loss:-8.6f} lr {lr:-.9f} memory {memory_usage:.1f}")
            if args.rank == 0:
                tb_writer.add_scalar("Loss", loss_list[-1], global_step)
            estimate_deviation(args, infer_model, tb_writer, global_step)
            loss_list = []

        # Log images
        if args.rank == 0 and args.logging_gen_steps > 0 and (global_step % args.logging_gen_steps == 0 or global_step in [1,100]):
            try: 
                test_batch_data=next(test_image_dataloader_iter)
            except Exception as e:
                test_image_dataloader_iter = iter(test_image_dataloader)
                test_batch_data=next(test_image_dataloader_iter)
            with torch.no_grad():
                nSample = min(args.train_batch_size, args.val_batch_size)
                # print(nSample)
                # pdb.set_trace()
                visualize(args, "images", test_batch_data, tb_writer, infer_model, global_step, nSample=nSample, nTest=1)
                # Sync to HDFS
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir, exist_ok=True)


        # Save models
        if args.save_steps > 0 and global_step % args.save_steps == 0:
            optimizer.consolidate_state_dict(to=0)
            if args.rank==0:
                print("Saving model checkpoint to %s", args.output_dir)
                models_state = dict()
                models_state[0] = infer_model.state_dict()
                if model_ema is not None:
                    models_state[args.ema_rate] = model_ema.ema_model.state_dict()
                save_checkpoint_ema(output_dir=args.output_dir,
                                models_state=models_state,
                                epoch=global_step,
                                optimizer_state=optimizer.state_dict(),
                                scaler_state=scaler.state_dict())
                print(f"model saved to {args.output_dir}")

        # Clear cache
        if first_print or global_step % 200 == 0:
            torch.cuda.empty_cache()
            print_peak_memory("Max memory allocated After running {} steps:".format(global_step), args.local_rank)

        first_print = False


if __name__ == "__main__":

    str2bool = lambda arg: bool(int(arg))
    parser = argparse.ArgumentParser(description='Control Net training')
    ## Model
    parser.add_argument('--model_config', type=str, default="model_lib/ControlNet/models/cldm_v15_video_appearance.yaml",
                        help="The path of model config file")
    parser.add_argument('--reinit_hint_block', action='store_true', default=False,
                        help="Re-initialize hint blocks for channel mis-match")
    parser.add_argument('--image_size',  type =int, default=64)
    parser.add_argument('--empty_text_prob', type=float, default=0.1,
                    help="For cfg, probablity of replacing text to empty seq ")
    parser.add_argument('--sd_locked', type =str2bool, default=True,
                        help='Freeze parameters in original stable-diffusion decoder')
    parser.add_argument('--only_mid_control', type =str2bool, default=False,
                        help='Only control middle blocks')
    parser.add_argument('--finetune_attn', action='store_true', default=False,
                        help='Fine-tune all self-attention layers in UNet')
    parser.add_argument('--finetune_control', action='store_true', default=False,
                        help='Fine-tune Control')
    parser.add_argument('--fix_hint', action='store_true', default=False,
                        help='fix hint layer in pose control')
    parser.add_argument('--finetune_pose_only', action='store_true', default=False,
                        help='Fine-tune Control')
    parser.add_argument('--finetune_reference_only', action='store_true', default=False,
                        help='Fine-tune Control')
    parser.add_argument('--finetune_mm', action='store_true', default=False,
                        help='Fine-tune motion module')
    parser.add_argument('--finetune_reference', action='store_true', default=False,
                        help='Fine-tune reference net, use together with arg finetune_mm')
    parser.add_argument('--finetune_mm_only', action='store_true', default=False,
                        help='Fine-tune motion module only and freeze all other params, use together with arg finetune_mm')
    parser.add_argument('--image_finetune', action='store_true', default=False,
                        help='Fine-tune control and freeze motion module, use together with arg finetune_mm')
    parser.add_argument('--image_finetune_unet', action='store_true', default=False,
                        help='Fine-tune unet and freeze motion module and control, use together with arg finetune_mm')
    parser.add_argument('--unet_pose_app', action='store_true', default=False,
                        help='Fine-tune unet and control and freeze motion module, use together with arg finetune_mm')
    parser.add_argument('--finetune_all', action='store_true', default=False,
                        help='Fine-tune all UNet and ControlNet parameters')
    parser.add_argument('--finetune_imagecond_unet', action='store_true', default=False,
                        help='Fine-tune all UNet and image ControlNet parameters')
    parser.add_argument('--control_type', type=str, nargs="+", default=["depth"],
                        help='The type of conditioning')
    parser.add_argument('--control_dropout', type=float, default=0.0,
                        help='The probability of dropping out control inputs, only applied for multi-control')
    parser.add_argument('--depth_bg_threshold', type=float, default=0.0,
                        help='The threshold of cutting off depth')
    parser.add_argument('--inpaint_unet', type=str2bool, default=False,
                        help='Train ControlNet for inpainting UNet')
    parser.add_argument('--blank_mask_prob', type=float, default=0.0,
                        help='Train ControlNet for inpainting UNet with blank mask')
    parser.add_argument('--mask_densepose', type=float, default=0.0,
                        help='Train ControlNet for with masked densepose (if used)')
    parser.add_argument("--control_mode", type=str, default="balance",
                        help="Set controlnet is more important or balance.")
    parser.add_argument('--wonoise', action='store_true', default=False,
                        help='Use with referenceonly, remove adding noise on reference image')
    ## Training
    parser.add_argument('--random_mask', action='store_true', default=False,
                        help='Randomly mask reference image. Similar to inpaint.')
    parser.add_argument('--num_workers', type = int, default = 4,
                        help='total number of workers for dataloaders')
    parser.add_argument('--train_batch_size', type = int, default = 16,
                        help='batch size for each gpu in distributed training')
    parser.add_argument('--val_batch_size', type = int, default = 1,
                        help='batch size for each gpu during inference(must be set to 1)')
    parser.add_argument('--lr', type = float, default = 1e-5,
                        help='learning rate of new params in control net')
    parser.add_argument('--lr_sd', type = float, default = 1e-5,
                        help='learning rate of params from original stable diffusion modules')
    parser.add_argument('--weight_decay', type = float, default = 0,
                        help='weight decay (L2) regularization')
    parser.add_argument('--lr_anneal_steps', type = float, default = 0,
                        help='steps for learning rate annealing')
    parser.add_argument('--ema_rate', type = float, default = 0,
                        help='rate for ema')
    parser.add_argument('--num_train_steps', type = int, default = 1000000,
                        help='number of train steps')
    parser.add_argument('--grad_clip_norm', type = float, default = 0.5,
                        help='grad_clip_norm')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--seed', type=int, default=42, 
                        help='random seed for initialization')
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--logging_gen_steps", type=int, default=1000, help="Log Generated Image every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=10000, help=" 10000 Save checkpoint every X updates steps.")
    parser.add_argument("--save_total_limit", type=int, default=100,
                        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default")
    parser.add_argument('--use_fp16', action='store_true', default=False,
                        help='Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit')
    parser.add_argument('--global_step', type=int, default=0,
                        help='initial global step to start with (use with --init_path)')
    parser.add_argument('--load_optimizer_state', type=str2bool, default=True,
                        help='Whether to restore optimizer when resuming')
    parser.add_argument('--compile', type=str2bool, default=False,
                        help='compile model (for torch 2)')
    parser.add_argument('--with_text', action='store_false', default=True,
                        help='Feed text_blip into the model')
    parser.add_argument('--v4', action='store_true', default=False,
                        help='dataset with original pose and image')
    parser.add_argument('--text_prompt', type=str, default=None,
                        help='Feed text_prompt into the model')
    parser.add_argument('--eta', type = float, default = 0.0,
                        help='eta during DDIM Sampling')
    parser.add_argument('--gif_time', type = float, default = 0.03,
                        help='gif per frame time')  
    ## Data
    parser.add_argument('--mask_bg', action='store_true', default=False,
                        help='Mask the background of image to pure black')
    parser.add_argument('--mask_face', action='store_true', default=False,
                        help='Mask the rest of face area to black')
    parser.add_argument("--train_dataset", type=str, default="laionhumanDs_densepose_1face_lm",
                        help="The dataset class for training.")
    parser.add_argument("--img_bin_limit", type = int, default = 29,
                        help="The upper limit while loading image from a sequence.")
    parser.add_argument("--frame_num", type = int, default = 16,
                        help="Number of frames in a single sequence of video.")
    parser.add_argument("--output_dir", type=str, default=None, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--local_log_dir", type=str, default=None, required=True,
                        help="The local output directory where tensorboard will be written.")
    parser.add_argument("--local_image_dir", type=str, default=None, required=True,
                        help="The local output directory where generated images will be written.")
    parser.add_argument("--resume_dir", type=str, default=None,
                        help="The resume directory where the model checkpoints will be loaded.")
    parser.add_argument("--image_pretrain_dir", type=str, default=None,
                        help="The resume directory where the appearance control model checkpoints will be loaded.")
    parser.add_argument("--pretrained_everything", type=str, default=None,
                        help="The directory where the appearance control and pose control and diffusion UNet model checkpoints will be loaded.")
    parser.add_argument("--mm_pretrain_dir", type=str, default=None,
                        help="The resume directory where the motion module checkpoints will be loaded.")
    parser.add_argument("--pose_pretrain_dir", type=str, default=None,
                        help="The resume directory where the pose control model checkpoints will be loaded.")
    parser.add_argument("--init_path", type=str, default="hdfs://harunava/home/byte_ailab_us_cvg/user/yichun.shi/pretrained/ControlNet/control_sd15_ini.ckpt",
                        help="The resume directory where the model checkpoints will be loaded.")
    args = parser.parse_args()

    main(args)