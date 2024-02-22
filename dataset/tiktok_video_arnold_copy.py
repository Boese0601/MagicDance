import os
import io

import random 
import json
import numpy as np
from base64 import b64decode
from PIL import Image, ImageStat
import pdb
import torch
from torchvision import transforms

from torch.utils.data import IterableDataset
from .safty import porn_list
from .mask import get_mask
from model_lib.ControlNet.annotator.openpose import util

from model_lib.ControlNet.annotator.util import resize_image, HWC3
from model_lib.ControlNet.annotator.openpose import OpenposeDetector
MONOCHROMATIC_MAX_VARIANCE = 0.3
from langdetect import detect
from model_lib.ControlNet.annotator.zoe import ZoeDetector

def is_english(text):
    try:
        lang = detect(text)
        return lang == 'en'
    except:
        return False
        

        
def draw_pose(pose, H, W, draw_body=True, draw_hand=True, draw_face=True):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if draw_body:
        canvas = util.draw_bodypose(canvas, candidate, subset)

    if draw_hand:
        canvas = util.draw_handpose(canvas, hands)

    if draw_face:
        canvas = util.draw_facepose(canvas, faces)

    return canvas


def is_monochromatic_image(pil_img):
    v = ImageStat.Stat(pil_img.convert('RGB')).var
    return sum(v)<MONOCHROMATIC_MAX_VARIANCE

def isnumeric(text):
    return (''.join(filter(str.isalnum, text))).isnumeric()

def meta_filter(meta_data):
    # True means the data is bad, we don't want the data
    if 'watermark_prob' in meta_data.keys() and meta_data['watermark_prob'] >= 0.5:
        return True 
    if 'clip_sim' in meta_data.keys() and meta_data['clip_sim'] < 0.25:
        return True 
    if 'aesthetic_score' in meta_data.keys() and meta_data['aesthetic_score'] <= 3.5:
        return True 
    if 'nsfw_score' in meta_data.keys() and meta_data['nsfw_score'] >= 0.93:
        return True 
    return False

def porn_filter(text):
    text = text.lower()
    for word in porn_list:
        word = word.lower()
        if word in text.split():
            return True 
        if len(word.split()) > 1 and word in text:
            return True
    return False

def filter_keypoints(image_size, candidate, subset, max_people=1, min_area=0.04):
    if len(subset) > max_people:
        return False
    for n in range(len(subset)):
        count = 0
        for i in range(18):
            index = int(subset[n][i])
            if index == -1:
                continue
            count += 1
        if count < 6:
            return False
    return True

class ImageTextControlDataset(IterableDataset): # pytorch rather than KV, replace all BYTES
    def __init__(self, data_path, pose_path, rank=0, world_size=1, shuffle=True, repeat=True, transform=None, lang_type="en", 
                control_type=None, mask_mode="free_form", with_pose=False, filter_pose=True, inpaint=False,train=True, img_bin_limit=29, random_mask=False, v4=False, pose_transform=None):
        super().__init__()
        assert len(data_path) > 0, "Data path must not be empty."
        assert len(pose_path) > 0, "Pose path must not be empty."
        assert rank < world_size and rank >= 0, "Rank must be >= 0 and < world_size."
        assert world_size >= 1, "World_size must be >= 1."
        self.data_path = data_path
        self.pose_path = pose_path
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.transform = transform
        self.lang_type = lang_type
        self.control_type = control_type or []
        self.mask_mode = mask_mode
        self.with_pose = with_pose
        self.filter_pose = filter_pose
        self.inpaint = inpaint
        self.debug = False
        self.train = train
        self.img_bin_limit = img_bin_limit
        self.random_mask = random_mask
        self.v4 = v4
        self.pose_transform = pose_transform
        self.without_face_prob = 0.2
        self.without_hand_prob = 0.5
        self.all_subjects = sorted(os.listdir(self.data_path))
        print(f"Creating ImageTextControlDataset rank={rank} world_size={world_size} data_path={data_path}")

    def __iter__(self):
        all_subjects = self.all_subjects
        random.shuffle(all_subjects)
        for subject_folder in all_subjects: 
            state = torch.get_rng_state()
            torch.set_rng_state(state)
            if self.train:
                try:
                    subject_folder_path = os.path.join(self.data_path, subject_folder)
                    subject_folder_images = os.listdir(subject_folder_path)  
                    subject_folder_images.sort()
                    subject_pose_path = os.path.join(self.pose_path, subject_folder)
                    subject_pose_maps = os.listdir(subject_pose_path)
                    subject_pose_maps.sort()
                    
                    if len(subject_folder_images) == 0 or len(subject_folder_images) == 1 or len(subject_pose_maps) == 0 or len(subject_pose_maps) == 1: # empty folder
                        print("empty source folder or folder with only one image detected")
                        continue
                    rand_int = torch.randint(low=0,high=len(subject_folder_images),size=(2,))
                    condition_int = rand_int[0].item()
                    input_int  = rand_int[1].item()
                    input_img_random_num = input_int % len(subject_folder_images)
                    input_pose_random_num = input_int % len(subject_pose_maps)
                    condition_img_random_num = condition_int % len(subject_folder_images)
                    condition_pose_random_num = condition_int % len(subject_pose_maps)
                    image_frame_path = os.path.join(subject_folder_path, subject_folder_images[input_img_random_num])
                    image_pil = Image.open(image_frame_path).convert("RGB")
                    W, H  = image_pil.size
                    condition_image_path = os.path.join(subject_folder_path, subject_folder_images[condition_img_random_num])
                    condition_image_pil = Image.open(condition_image_path).convert("RGB")
                    if is_monochromatic_image(condition_image_pil):
                        continue
                    if self.transform is not None:
                        condition_image = self.transform(condition_image_pil)
                    if condition_image.std() < 0.02:
                        continue

                    if is_monochromatic_image(image_pil):
                        continue
                    if self.transform is not None:
                        image = self.transform(image_pil)
                    if image.std() < 0.02:
                        continue



                    res = {'condition_image': condition_image, 'image': image}
                    

                    if self.mask_mode is not None and self.random_mask:
                        randommask = get_mask(mask_mode=self.mask_mode, img_size=condition_image.shape[1:])
                        res["randommask"] = randommask
                        
                    if not self.v4:
                        if self.with_pose:
                            if "body+hand+face" in self.control_type: 
                                pose_map_path = os.path.join(subject_pose_path, subject_pose_maps[input_pose_random_num])
                                pose_map = Image.open(pose_map_path).convert("RGB")
                                pose_map = self.pose_transform(pose_map)
                                print("pose map type:",type(pose_map))
                                res["pose_map"] = pose_map
                                src_pose_map_path = os.path.join(subject_pose_path, subject_pose_maps[condition_pose_random_num])
                                src_pose_map = Image.open(src_pose_map_path).convert("RGB")
                                src_pose_map = self.pose_transform(src_pose_map)
                                print("src pose map type:",type(src_pose_map))
                                res["src_pose_map"] = src_pose_map
                            else: 
                                raise ValueError("Invalid control_type")
                    else:
                        if "depth" in self.control_type:
                            raise ValueError("Depth map is not supported yet.")
                        else:
                            pose_map_path = os.path.join(subject_pose_path, subject_pose_maps[input_pose_random_num])
                            pose_pil = Image.open(pose_map_path).convert("RGB")
                            if self.pose_transform is not None:
                                pose = self.pose_transform(pose_pil)
                                    # condition_pose = condition_pose.float() / 255.0
                            res["pose_map"] = pose
                            cond_pose_map_path = os.path.join(subject_pose_path, subject_pose_maps[condition_pose_random_num])
                            condition_pose_pil = Image.open(cond_pose_map_path).convert("RGB")
                            if self.pose_transform is not None:
                                condition_pose = self.pose_transform(condition_pose_pil)
                                    # condition_pose = condition_pose.float() / 255.0
                            res["src_pose_map"] = condition_pose
                    yield res

                except Exception as e:
                    raise(e)
                    
            else: # inference 
                # print("infer folder:",subject_folder)
                subject_folder_path = os.path.join(self.data_path, subject_folder)
                subject_folder_images = os.listdir(subject_folder_path)  
                subject_folder_images.sort()
                subject_pose_path = os.path.join(self.pose_path, subject_folder)
                subject_pose_maps = os.listdir(subject_pose_path)
                subject_pose_maps.sort()
                
                condition_int = 0
                condition_image_path = os.path.join(subject_folder_path, subject_folder_images[condition_int])
                condition_image_pil = Image.open(condition_image_path).convert("RGB")
                W, H  = condition_image_pil.size
                
                if is_monochromatic_image(condition_image_pil):
                    continue
                if self.transform is not None:
                    condition_image = self.transform(condition_image_pil)
                if condition_image.std() < 0.02:
                    continue

                res = {'condition_image': condition_image}
                
                if self.mask_mode is not None and self.random_mask:
                    randommask = get_mask(mask_mode=self.mask_mode, img_size=condition_image.shape[1:])
                    res["randommask"] = randommask
                    
                if "depth" in self.control_type:
                    pass
                else:
                    condition_pose_path = os.path.join(subject_pose_path, subject_pose_maps[condition_int])
                    condition_pose_pil = Image.open(condition_pose_path).convert("RGB")
                    if self.pose_transform is not None:
                        condition_pose = self.pose_transform(condition_pose_pil)
                        # condition_pose = condition_pose.float() / 255.0
                    res["src_pose_map"] = condition_pose

                image_list = []
                pose_map_list = []
                if self.img_bin_limit != 'all' :
                    image_range = min(int(self.img_bin_limit),len(subject_folder_images))
                else:
                    image_range = len(subject_folder_images)
                    
                for i in range(image_range-1):

                    image_frame_path = os.path.join(subject_folder_path, subject_folder_images[i+1])
                    image_pil = Image.open(image_frame_path).convert("RGB")
                    if is_monochromatic_image(image_pil):
                        continue
                    if self.transform is not None:
                        image = self.transform(image_pil)

                    if image.std() < 0.02:
                        continue
                    image_list.append(image)
                    pose_frame_path = os.path.join(subject_pose_path, subject_pose_maps[i+1])
                    pose_pil = Image.open(pose_frame_path).convert("RGB")
                    if self.pose_transform is not None:
                        pose_map = self.pose_transform(pose_pil)
                    pose_map_list.append(pose_map)
                res['image_list'] = image_list
                res['pose_map_list'] = pose_map_list
                yield res


def init_ds(train_data_path, train_pose_path, *args, **kwargs):
    return ImageTextControlDataset(train_data_path, train_pose_path, *args, **kwargs)


def tiktok_video_arnold(**kwargs):
    train_data_path = "./TikTok-v4/train_set"
    train_pose_path = "./TikTok-v4/pose_map_train_set"
    return init_ds(train_data_path, train_pose_path, with_pose=True, **kwargs)


def tiktok_video_arnold_val(**kwargs):
    data_path = "./TikTok-v4/disco_test_set"
    pose_path = "./TikTok-v4/pose_map_disco_test_set"
    return init_ds(data_path, pose_path, with_pose=True, **kwargs)

