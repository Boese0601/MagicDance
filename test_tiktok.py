"""
Train a control net
"""

import os
import sys
import argparse
import datetime
import numpy as np
import pdb

# # torch
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torchvision import transforms as T
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# #from ema_pytorch import EMA
# import kornia

# # distributed 
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP 

# # data
from dataset import  tiktok_video_arnold_copy
from dataset.hdfs_io import hcopy, hexists
from dataset.transforms import RemoveWhite, CenterCrop

# # utils
from utils.checkpoint import load_from_pretrain, save_checkpoint_ema
from utils.utils import set_seed, count_param, print_peak_memory, anal_tensor
from utils.lr_scheduler import LambdaLinearScheduler
from dataset.hdfs_io import hexists, hmkdir, hopen, hcopy
from langdetect import detect
# # model
from model_lib.ControlNet.cldm.model import create_model, instantiate_from_config
import copy

import torchvision.transforms as transforms
from PIL import Image
import imageio


def center_crop_to_512(image_path):
    # Read the image from the local path
    image = Image.open(image_path)
    # Define the transformation to center crop to 512x512
    transform = transforms.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(
                    512,
                    scale=(1.0, 1.0), ratio=(1., 1.),
                    interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(), 
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Apply the transformation
    cropped_image = transform(image)

    return cropped_image

def center_crop_pose_to_512(image_path):
    # Read the image from the local path
    image = Image.open(image_path)

    # Define the transformation to center crop to 512x512
    transform =  T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.RandomResizedCrop(
                512,
                scale=(1.0, 1.0), ratio=(1., 1.),
                interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(), 
    ])
    # Apply the transformation
    cropped_image = transform(image)

    return cropped_image


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

TORCH_VERSION = torch.__version__.split(".")[0]
FP16_DTYPE = torch.float16 if TORCH_VERSION == "1" else torch.bfloat16
print(f"TORCH_VERSION={TORCH_VERSION} FP16_DTYPE={FP16_DTYPE}")

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


def load_state_dict(model, ckpt_path, reinit_hint_block=False, strict=True, map_location="cpu"):
    print(f"Loading model state dict from {ckpt_path} ...")
    state_dict = load_from_pretrain(ckpt_path, map_location=map_location)
    state_dict = state_dict.get('state_dict', state_dict)
    # print(state_dict.keys()) # check the keys of pythhon dict. if sdon't work set strict=false in the call VV 466
    if reinit_hint_block:
        print("Ignoring hint block parameters from checkpoint!")
        for k in list(state_dict.keys()):
            if k.startswith("control_model.input_hint_block"):
                state_dict.pop(k)
    model.load_state_dict(state_dict, strict=strict)
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

def get_cond_control(args, batch_data, control_type, device, model=None, batch_size=None, train=True, pose_transfer=False, batch_data_2=None):
    # Single-control
    control_type = copy.deepcopy(control_type)[0]
    if control_type == "body+hand+face" :
        if train:
            raise NotImplementedError
        else:
            # if pose_transfer:
            #     assert "pose_map_list" in batch_data_2
            #     pose_map_list = batch_data_2["pose_map_list"]
            # else:
            #     assert "pose_map_list" in batch_data
            #     pose_map_list = batch_data["pose_map_list"]
            # print("Get Inference Control Type")
            
            if args.local_cond_image_path is not None:
                cond_image = center_crop_to_512(args.local_cond_image_path).unsqueeze(0).cuda()
            else:
                cond_image = batch_data["condition_image"].cuda()
            # pdb.set_trace()
            if args.local_pose_path is not None:
                #print("local pose path:",args.local_pose_path)
                pose_map_list = []
                for pose_map_name in sorted(os.listdir(args.local_pose_path)):
                    pose_map = center_crop_pose_to_512(os.path.join(args.local_pose_path, pose_map_name)).unsqueeze(0)
                    pose_map_list.append(pose_map)
            else:
                #print("no local pose path")
                #print(batch_data["pose_map_list"])
                pose_map_list = batch_data["pose_map_list"]
            c_cat_list = [pose_map for pose_map in pose_map_list]
            cond_image = model.get_first_stage_encoding(model.encode_first_stage(cond_image))
            cond_img_cat = cond_image
    else:
        raise NotImplementedError(f"cond_type={control_type} not supported!")

    if train:
        raise NotImplementedError
    else:
        return_list = []
        #print(len(c_cat_list))
        for c_cat in c_cat_list:
            #print(c_cat)
            if args.control_dropout > 0:
                mask = torch.rand((c_cat.shape[0],1,1,1)) > args.control_dropout
                c_cat = c_cat * mask.type(torch.float32).to(device)
            # pdb.set_trace()
            return_list.append([c_cat[:batch_size]])
        # pdb.set_trace()
        return return_list, [cond_img_cat[:batch_size].to(device) ]

def visualize(args, name, batch_data, tb_writer, infer_model, global_step, nSample, nTest=1, pose_transfer=False, batch_data_2=None):
    gen_image_path = os.path.join(args.local_image_dir,str(global_step),'gen_images')
    gt_image_path = os.path.join(args.local_image_dir,str(global_step),'gt_images')
    pose_map_path = os.path.join(args.local_image_dir,str(global_step),'pose_maps')
    os.makedirs(gen_image_path,exist_ok=True)
    os.makedirs(gt_image_path,exist_ok=True)
    os.makedirs(pose_map_path,exist_ok=True)
    infer_model.eval()

    if args.pose_transfer:
        real_image_list = [real_image for real_image in batch_data_2["image_list"]]
    else:
        real_image_list = [real_image for real_image in batch_data["image_list"]]
    # pdb.set_trace()
    if not args.v4:
        # text = batch_data["text_blip"][:nSample]
        text = [""] * nSample
    else:
        text = [""] * nSample
    if not args.with_text:
        for i in range(len(text)):
            text[i] = ""
    if args.text_prompt is not None: 
        for i in range(len(text)):
            text[i] = args.text_prompt
    
    #print("batch data:")
    #print(batch_data)
    
    c_cat_list, cond_img_cat = get_cond_control(args, batch_data, args.control_type, args.device, model=infer_model, batch_size=nSample, train=False, pose_transfer=pose_transfer, batch_data_2=batch_data_2)
    c_cross = infer_model.get_learned_conditioning(text)[:nSample]
    uc_cross = infer_model.get_unconditional_conditioning(nSample)
    noise_shape = (nSample, infer_model.channels, infer_model.image_size, infer_model.image_size)
    noise = torch.randn(noise_shape).cuda()
    if args.local_cond_image_path is not None:
        cond_image = center_crop_to_512(args.local_cond_image_path).unsqueeze(0)
    else:
        cond_image = batch_data["condition_image"]
    cond_grid = make_grid(cond_image.float().clamp(-1,1).cpu().add(1).mul(0.5), nrow=1)
    save_image(cond_grid, os.path.join(args.local_image_dir,str(global_step),"condition.jpg"))
    for img_num in range(len(c_cat_list)):
        real_image = real_image_list[img_num][:nSample].cuda()
        pose_input = c_cat_list[img_num][0].cuda()
        print("Generate Image {} in {} images".format(img_num,len(c_cat_list))) 
        c = {"c_concat": [pose_input], "c_crossattn": [c_cross], "image_control":cond_img_cat}
        if args.control_mode == "controlnet_important":
            uc = {"c_concat": [pose_input], "c_crossattn": [uc_cross]}
        else:
            uc = {"c_concat": [pose_input], "c_crossattn": [uc_cross], "image_control":cond_img_cat}
        if args.wonoise:
            c['wonoise'] = True
            uc['wonoise'] = True
        else:
            c['wonoise'] = False
            uc['wonoise'] = False
        c['overlap_sampling'] = False
        uc['overlap_sampling'] = False
        if args.inpaint_unet:
            inpaint, masked_image = get_cond_inpaint(infer_model, batch_data, args.device, batch_size=nSample, blank_mask=False)
            inpaint_list = [masked_image.cpu()]
        else:
            inpaint = None
            inpaint_list = []

        # generate images
        with torch.cuda.amp.autocast(enabled=args.use_fp16, dtype=FP16_DTYPE): # FP16_DTYPE):
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
            next_cond = copy.deepcopy(gene_img)


        latent = infer_model.get_first_stage_encoding(infer_model.encode_first_stage(real_image))
        rec_image = infer_model.decode_first_stage(latent)
        gen_grid = make_grid(gene_img.float().clamp(-1,1).cpu().add(1).mul(0.5), nrow=1)
        rec_grid = make_grid(rec_image.float().clamp(-1,1).cpu().add(1).mul(0.5), nrow=1)
        pose_grid = make_grid(c_cat_list[img_num][0][:,:3].float().clamp(-1,1).cpu().add(1).mul(0.5), nrow=1)
        save_image(gen_grid, os.path.join(gen_image_path,f"{img_num:03d}.jpg"))
        save_image(rec_grid, os.path.join(gt_image_path,f"{img_num:03d}.jpg"))
        save_image(pose_grid, os.path.join(pose_map_path,f"{img_num:03d}.jpg"))




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


    # ******************************
    # load pre-trained models
    # ******************************
    optimizer_state_dict = None
    global_step = args.global_step

    if args.image_pretrain_dir.endswith(".th"):
        print('find model state dict from {} ...'.format(args.image_pretrain_dir))
        # if still doesn't work with strict=false, comment out and test rest
        if args.local_rank == 0:
            assert hexists(args.image_pretrain_dir)
            load_state_dict(model, args.image_pretrain_dir,strict=False) # if strict=true, want all checkpoint weights to be exactly the same as model
    else:
        print('find optimizer state dict from {} ...'.format(os.path.join(args.image_pretrain_dir, "optimizer_state_latest.th")))
        optimizer_state_dict = load_from_pretrain(os.path.join(args.image_pretrain_dir, "optimizer_state_latest.th"), map_location="cpu")
        if global_step == 0:
            global_step = optimizer_state_dict["step"]
        optimizer_state_dict = None
        if args.local_rank == 0:
            assert hexists(os.path.join(args.image_pretrain_dir, f"model_state-{global_step}.th"))
            load_state_dict(model, os.path.join(args.image_pretrain_dir, f"model_state-{global_step}.th"),strict=True)
    global_step = 0
    torch.cuda.empty_cache()


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
            test_image_transform =  T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(
                    args.image_size*8,
                    scale=(1.0, 1.0), ratio=(1., 1.),
                    interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(), 
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
            test_pose_transform =  T.Compose([
            T.RandomResizedCrop(
                    args.image_size*8,
                    scale=(1.0, 1.0), ratio=(1., 1.),
                    interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(), 
        ])
    else:
        test_image_transform =  T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.RandomResizedCrop(
                args.image_size*8,
                scale=(1.0, 1.0), ratio=(1., 1.),
                interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(), 
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
        print("tiktok_video_arnold")
        test_dataset_cls = getattr(tiktok_video_arnold_copy, args.train_dataset+'_val')
        test_image_dataset = test_dataset_cls(transform=test_image_transform,
                                rank=args.rank,
                                world_size=args.world_size,
                                control_type=args.control_type,
                                inpaint=args.inpaint_unet,
                                train=False,
                                img_bin_limit=args.img_bin_limit,
                                random_mask=False,
                                v4=args.v4,
                                pose_transform=test_pose_transform,
                                )
    else:
        raise NotImplementedError

    test_image_dataloader = DataLoader(test_image_dataset, 
                                  batch_size=args.val_batch_size,
                                  num_workers=1,
                                  pin_memory=True)
    test_image_dataloader_iter = iter(test_image_dataloader)
    # print("test_image_iter:",test_image_dataloader_iter)
    
    if args.local_rank == 0:
        print(f"image dataloader created: dataset={args.train_dataset} batch_size={args.train_batch_size} control_type={args.control_type}")

    dist.barrier()
    
    #-
    # local_step = 0
    # loss_list = []
    #-
    
    #first_print = True
    infer_model = model.module if hasattr(model, "module") else model
    print(f"[rank{args.rank}] start training loop!")
    
    estimate_deviation(args, infer_model, tb_writer, global_step)
    
    print("train steps:",args.num_train_steps)
    
    for itr in range(0, args.num_train_steps):
            #-
            # Get input
            # batch_data = next(image_dataloader_iter)
            #-
            
        #print("in the loop!")
            
        test_batch_data=next(test_image_dataloader_iter)
        #print("test batch data:",len(test_batch_data))
        os.makedirs(os.path.join(args.local_image_dir,str(itr)),exist_ok=True)
        with torch.no_grad():
            #print("args:",args)
            nSample = min(args.train_batch_size, args.val_batch_size)
            visualize(args, "val_images", test_batch_data, tb_writer, infer_model, itr, nSample=nSample, nTest=1, pose_transfer=args.pose_transfer, batch_data_2=None)

        # if first_print or itr % 200 == 0:
        #     torch.cuda.empty_cache()
        #     print_peak_memory("Max memory allocated After running {} steps:".format(itr), args.local_rank)

        # first_print = False


if __name__ == "__main__":

    str2bool = lambda arg: bool(int(arg))
    parser = argparse.ArgumentParser(description='Control Net training')
    ## Model
    parser.add_argument('--model_config', type=str, default="model_lib/ControlNet/models/cldm_v15_video.yaml",
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
    parser.add_argument('--finetune_all', action='store_true', default=False,
                        help='Fine-tune all UNet and ControlNet parameters')
    parser.add_argument('--finetune_imagecond_unet', action='store_true', default=False,
                        help='Fine-tune all UNet and image ControlNet parameters')
    parser.add_argument('--control_type', type=str, nargs="+", default=["pose"],
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
    parser.add_argument('--mask_bg', action='store_true', default=False,
                        help='Mask the background of image to pure black')
    ## Training
    parser.add_argument("--img_bin_limit", default = 29,
                        help="The upper limit while loading image from a sequence.")
    parser.add_argument('--num_workers', type = int, default = 1,
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
    parser.add_argument('--pose_transfer', action='store_true', default=False,
                        help='Mode: pose_transfer, default: Self reconstruction')
    parser.add_argument('--eta', type = float, default = 0.0,
                        help='eta during DDIM Sampling')
    parser.add_argument('--autoreg', action='store_true', default=False,
                        help='Auto Regressively generate result')
    parser.add_argument('--gif_time', type = float, default = 0.03,
                        help='gif per frame time')  
    parser.add_argument('--text_prompt', type=str, default=None,
                        help='Feed text_prompt into the model')         
    ## Data
    parser.add_argument('--v4', action='store_true', default=False,
                        help='dataset with original pose and image')
    parser.add_argument("--train_dataset", type=str, default="laionhumanDs_densepose_1face_lm",
                        help="The dataset class for training.")
    parser.add_argument("--output_dir", type=str, default=None, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--local_log_dir", type=str, default=None, required=False,
                        help="The local output directory where tensorboard will be written.")
    parser.add_argument("--local_image_dir", type=str, default=None, required=True,
                        help="The local output directory where generated images will be written.")
    parser.add_argument("--resume_dir", type=str, default=None,
                        help="The resume directory where the model checkpoints will be loaded.")
    parser.add_argument("--image_pretrain_dir", type=str, default=None,
                        help="The resume directory where the model checkpoints will be loaded.")
    parser.add_argument("--pose_pretrain_dir", type=str, default=None,
                        help="The resume directory where the model checkpoints will be loaded.")
    parser.add_argument("--init_path", type=str, default="hdfs://harunava/home/byte_ailab_us_cvg/user/yichun.shi/pretrained/ControlNet/control_sd15_ini.ckpt",
                        help="The resume directory where the model checkpoints will be loaded.")
    parser.add_argument('--local_cond_image_path', type=str, default=None, help='Cond image')
    parser.add_argument('--local_pose_path', type=str, default=None, help='Pose maps')
    args = parser.parse_args()

    print("Customize text prompt:",args.text_prompt)
    main(args)