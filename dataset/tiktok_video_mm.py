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

from .kv_dataset import KVDataset
from .safty import porn_list
from .mask import get_mask
from model_lib.ControlNet.annotator.openpose import util

from model_lib.ControlNet.annotator.util import resize_image, HWC3
from model_lib.ControlNet.annotator.openpose import OpenposeDetector
MONOCHROMATIC_MAX_VARIANCE = 0.3
from langdetect import detect
# import euler
# import thriftpy2


# class SegInference(object):
#     def __init__(self):
#         module_path = os.path.dirname(__file__)
#         self.base_thrift = thriftpy2.load(
#             '/mnt/bn/dichang-bytenas/CelebV-Text/code/facelib/facelib/seg/idls/base.thrift', 
#             module_name='base_thrift'
#         )
#         self.server_thrift = thriftpy2.load(
#             '/mnt/bn/dichang-bytenas/CelebV-Text/code/facelib/facelib/seg/idls/iccv/segment_inference.thrift', 
#             module_name='seg_inference_thrift'
#         )
#         self.client = euler.Client(
#             self.server_thrift.SegmentInferenceService, 
#             'sd://ic.cv.segment_inference?cluster=default&idc=lf', 
#             timeout=60, disable_logging=True)

#     def predict(self, img, req_type="human_seg", n_attempt=10):
#         buffer = io.BytesIO()
#         img.save(buffer, format='PNG')
#         img_data = buffer.getvalue()

#         req = self.server_thrift.SegmentRequest()
#         req.image = img_data
#         if isinstance(req_type, list):
#             req.req_type_list = req_type
#         else:
#             req.req_type_list = [req_type]

#         for i in range(n_attempt):
#             try:
#                 rsp = self.client.Process(req)
#                 if isinstance(req_type, list):
#                     masks = [np.array(Image.open(io.BytesIO(rsp.results[r]))) for r in req_type]
#                     return tuple(masks)
#                 else:
#                     # mask = np.array(Image.open(io.BytesIO(rsp.results[req_type])))
#                     mask = Image.open(io.BytesIO(rsp.results[req_type]))
#                     return mask
#             except Exception as e:
#                 if i < n_attempt-1:
#                     continue
#                 else:
#                     raise(e)

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

class ImageTextControlDataset(KVDataset):
    def __init__(self, data_path, rank=0, world_size=1, shuffle=True, repeat=True, transform=None, lang_type="en", 
                control_type=None, mask_mode="free_form", with_pose=False, filter_pose=True, inpaint=False,train=True, frame_num=16, random_mask=False, v4=False, pose_transform=None):
        super().__init__(data_path, rank, world_size, shuffle, repeat)
        self.transform = transform
        self.lang_type = lang_type
        self.control_type = control_type or []
        self.mask_mode = mask_mode
        self.with_pose = with_pose
        self.filter_pose = filter_pose
        self.inpaint = inpaint
        self.debug = False
        self.train = train
        self.frame_num = frame_num
        self.random_mask = random_mask
        self.v4 = v4
        self.pose_transform = pose_transform
        print(f"Creating ImageTextControlDataset rank={rank} world_size={world_size} data_path={data_path}")

    def __iter__(self):
        for example in super().__iter__():
            state = torch.get_rng_state()
            torch.set_rng_state(state)
            if self.train:
                try:                    
                    data_item = json.loads(example)
                    img_keys = data_item.keys()
                    if len(img_keys)-self.frame_num-1 < 1:
                        continue
                    condition_int = torch.randint(low=0,high=len(img_keys)-self.frame_num-1,size=(1,))
                    # print("c_int",condition_int)
                    condition_image_item = data_item[list(img_keys)[condition_int]]
                    condition_image_str = b64decode(condition_image_item.get("img_str", condition_image_item.get("binary", condition_image_item.get("b64_binary", condition_image_item.get("image","")))))
                    condition_image_pil = Image.open(io.BytesIO(condition_image_str)).convert("RGB")
                    condition_mask_str = b64decode(condition_image_item.get("mask_str", condition_image_item.get("binary", condition_image_item.get("b64_binary", condition_image_item.get("image","")))))
                    condition_mask_pil = Image.open(io.BytesIO(condition_mask_str)).convert("RGB")
                    if is_monochromatic_image(condition_image_pil):
                        continue
                    if self.transform is not None:
                        condition_image = self.transform(condition_image_pil)
                    if condition_image.std() < 0.02:
                        continue
 
                    if self.transform is not None:
                        condition_mask = self.transform(condition_mask_pil)


                    if not self.v4:
                        if "text_blip" in condition_image_item:
                            text_blip = condition_image_item["text_blip"]
                        else:
                            pass
                        
                        if "text_bg" in condition_image_item:
                            text_bg = condition_image_item["text_bg"]
                        else:
                            pass

                        if len(text_bg) > 0 and text_bg[0] == ',': 
                            text_bg = text_bg[1:]
                        if isnumeric(text_bg):
                            continue
                        if porn_filter(text_bg):
                            continue
                        if not is_english(text_bg):
                            continue
                        # Load text from blip-v2
                        if len(text_blip) > 0 and text_blip[0] == ',': 
                            text_blip = text_blip[1:]
                        if text_blip==None or len(text_blip) == 0:
                            print('empty text_blip input: %s, will skip' % text_blip)
                            continue
                        if isnumeric(text_blip):
                            continue
                        if porn_filter(text_blip):
                            continue
                        if not is_english(text_blip):
                            continue

                        # res = {'condition_image': condition_image, 'face_mask': head_tensor, 'text_blip': text_blip, 'text_bg': text_bg, 'condition_mask': condition_mask}
                        res = {'condition_image': condition_image, 'text_blip': text_blip, 'text_bg': text_bg, 'condition_mask': condition_mask}
                    
                    else:
                        res = {'condition_image': condition_image, 'condition_mask': condition_mask}

                    if self.mask_mode is not None and self.random_mask:
                        randommask = get_mask(mask_mode=self.mask_mode, img_size=condition_image.shape[1:])
                        res["randommask"] = randommask
                        
                    H, W = condition_image.shape[1:]
                    if not self.v4:
                        if self.with_pose:
                            if "body+hand+face" in self.control_type: 
                    
                                src_pose = condition_image_item["openpose"]
                                src_pose_map = draw_pose(src_pose,H,W)
        
                                src_pose_map = torch.from_numpy(src_pose_map.copy()).float() / 255.0
                                src_pose_map = src_pose_map.permute(2,0,1)
                                res["src_pose_map"] = src_pose_map

                            elif "body+face" in self.control_type:

                                src_pose = condition_image_item["openpose"]
                                src_pose_map = draw_pose(src_pose,H,W,draw_hand=False)
                                src_pose_map = torch.from_numpy(src_pose_map.copy()).float() / 255.0
                                src_pose_map = src_pose_map.permute(2,0,1)
                                res["src_pose_map"] = src_pose_map

                            elif "body" in self.control_type:

                                src_pose = condition_image_item["openpose"]
                                src_pose_map = draw_pose(src_pose,H,W,draw_hand=False,draw_face=False)
                                src_pose_map = torch.from_numpy(src_pose_map.copy()).float() / 255.0
                                src_pose_map = src_pose_map.permute(2,0,1)
                                res["src_pose_map"] = src_pose_map
                    else:
                        condition_pose_str = b64decode(condition_image_item.get("pose_str", condition_image_item.get("binary", condition_image_item.get("b64_binary", condition_image_item.get("image","")))))
                        # print('condition_pose_str:',condition_pose_str)
                        
                        condition_pose_pil = Image.open(io.BytesIO(condition_pose_str)).convert("RGB")
                        if self.pose_transform is not None:
                            condition_pose = self.pose_transform(condition_pose_pil)
                            # condition_pose = condition_pose.float() / 255.0
                        res["src_pose_map"] = condition_pose

                    
                    # pdb.set_trace()
                    # TODO here, video dataloader
                    image_list = []
                    pose_map_list = []
                    mask_list = []

                    image_range = self.frame_num
                    if image_range > len(img_keys):
                        continue
                    for i in range(image_range):
                        # print("int",condition_int+i+1)
                        start_int = torch.randint(low=0,high=len(img_keys)-self.frame_num-1,size=(1,))

                        image_item = data_item[list(img_keys)[start_int+i]]

                        image_str = b64decode(image_item.get("img_str", image_item.get("binary", image_item.get("b64_binary", image_item.get("image","")))))
                        image_pil = Image.open(io.BytesIO(image_str)).convert("RGB")
                        mask_str = b64decode(image_item.get("mask_str", image_item.get("binary", image_item.get("b64_binary", image_item.get("image","")))))
                        mask_pil = Image.open(io.BytesIO(mask_str)).convert("RGB")



                        if is_monochromatic_image(image_pil):
                            continue
                        if self.transform is not None:
                            image = self.transform(image_pil)
                        if image.std() < 0.02:
                            continue

                        if self.transform is not None:
                            mask = self.transform(mask_pil)
                        
                        if not self.v4:
                            if self.filter_pose:
                                assert "openpose" in image_item
                                candidate = np.array(image_item["openpose"]["bodies"]["candidate"])
                                subset = np.array(image_item["openpose"]["bodies"]["subset"])
                                if not filter_keypoints(image.shape[:2], candidate, subset):
                                    continue

                        image_list.append(image)
                        mask_list.append(mask)
                        H, W = image.shape[1:]
                        
                        if not self.v4:
                            if self.with_pose:
                                if "body+hand+face" in self.control_type: 
                                    pose = image_item["openpose"]
                                    pose_map = draw_pose(pose,H,W)
            
                                    pose_map = torch.from_numpy(pose_map.copy()).float() / 255.0
                                    pose_map = pose_map.permute(2,0,1)
                                    pose_map_list.append(pose_map)
                                elif "body+face" in self.control_type:
                                    pose = image_item["openpose"]
                                    pose_map = draw_pose(pose,H,W,draw_hand=False)
                                    pose_map = torch.from_numpy(pose_map.copy()).float() / 255.0
                                    pose_map = pose_map.permute(2,0,1)
                                    pose_map_list.append(pose_map)
                                elif "body" in self.control_type:
                                    pose = image_item["openpose"]
                                    pose_map = draw_pose(pose,H,W,draw_hand=False,draw_face=False)
                                    pose_map = torch.from_numpy(pose_map.copy()).float() / 255.0
                                    pose_map = pose_map.permute(2,0,1)
                                    pose_map_list.append(pose_map)
                        else:
                            pose_str = b64decode(image_item.get("pose_str", image_item.get("binary", image_item.get("b64_binary", image_item.get("image","")))))
                            pose_pil = Image.open(io.BytesIO(pose_str)).convert("RGB")
                            if self.pose_transform is not None:
                                pose_map = self.pose_transform(pose_pil)
                                # pose_map = pose_map.float() / 255.0
                            pose_map_list.append(pose_map)
                    res['image_list'] = torch.stack(image_list)
                    res['pose_map_list'] = torch.stack(pose_map_list)
                    res['mask_list'] = torch.stack(mask_list)
                    yield res

                except Exception as e:
                    raise(e)

            else:
                if self.frame_num == "all": # TODO load all frames
                    try:                    
                        data_item = json.loads(example)
                        img_keys = data_item.keys()
                        # if len(img_keys)-self.frame_num-1 < 1:
                        #     continue
                        condition_int = 0
                        condition_image_item = data_item[list(img_keys)[condition_int]]
                        condition_image_str = b64decode(condition_image_item.get("img_str", condition_image_item.get("binary", condition_image_item.get("b64_binary", condition_image_item.get("image","")))))
                        condition_image_pil = Image.open(io.BytesIO(condition_image_str)).convert("RGB")
                        condition_mask_str = b64decode(condition_image_item.get("mask_str", condition_image_item.get("binary", condition_image_item.get("b64_binary", condition_image_item.get("image","")))))
                        condition_mask_pil = Image.open(io.BytesIO(condition_mask_str)).convert("RGB")
                        if is_monochromatic_image(condition_image_pil):
                            continue
                        if self.transform is not None:
                            condition_image = self.transform(condition_image_pil)
                        if condition_image.std() < 0.02:
                            continue
    
                        if self.transform is not None:
                            condition_mask = self.transform(condition_mask_pil)

                        if not self.v4:
                            if "text_blip" in condition_image_item:
                                text_blip = condition_image_item["text_blip"]
                            else:
                                pass
                            
                            if "text_bg" in condition_image_item:
                                text_bg = condition_image_item["text_bg"]
                            else:
                                pass

                            if len(text_bg) > 0 and text_bg[0] == ',': 
                                text_bg = text_bg[1:]
                            if isnumeric(text_bg):
                                continue
                            if porn_filter(text_bg):
                                continue
                            if not is_english(text_bg):
                                continue
                            # Load text from blip-v2
                            if len(text_blip) > 0 and text_blip[0] == ',': 
                                text_blip = text_blip[1:]
                            if text_blip==None or len(text_blip) == 0:
                                print('empty text_blip input: %s, will skip' % text_blip)
                                continue
                            if isnumeric(text_blip):
                                continue
                            if porn_filter(text_blip):
                                continue
                            if not is_english(text_blip):
                                continue

                            # res = {'condition_image': condition_image, 'face_mask': head_tensor, 'text_blip': text_blip, 'text_bg': text_bg, 'condition_mask': condition_mask}
                            res = {'condition_image': condition_image, 'text_blip': text_blip, 'text_bg': text_bg, 'condition_mask': condition_mask}
                        else:
                            res = {'condition_image': condition_image, 'condition_mask': condition_mask}



                        if self.mask_mode is not None and self.random_mask:
                            randommask = get_mask(mask_mode=self.mask_mode, img_size=condition_image.shape[1:])
                            res["randommask"] = randommask
                            
                        H, W = condition_image.shape[1:]
                        if not self.v4:
                            if self.with_pose:
                                if "body+hand+face" in self.control_type: 
                        
                                    src_pose = condition_image_item["openpose"]
                                    src_pose_map = draw_pose(src_pose,H,W)
            
                                    src_pose_map = torch.from_numpy(src_pose_map.copy()).float() / 255.0
                                    src_pose_map = src_pose_map.permute(2,0,1)
                                    res["src_pose_map"] = src_pose_map

                                elif "body+face" in self.control_type:

                                    src_pose = condition_image_item["openpose"]
                                    src_pose_map = draw_pose(src_pose,H,W,draw_hand=False)
                                    src_pose_map = torch.from_numpy(src_pose_map.copy()).float() / 255.0
                                    src_pose_map = src_pose_map.permute(2,0,1)
                                    res["src_pose_map"] = src_pose_map

                                elif "body" in self.control_type:

                                    src_pose = condition_image_item["openpose"]
                                    src_pose_map = draw_pose(src_pose,H,W,draw_hand=False,draw_face=False)
                                    src_pose_map = torch.from_numpy(src_pose_map.copy()).float() / 255.0
                                    src_pose_map = src_pose_map.permute(2,0,1)
                                    res["src_pose_map"] = src_pose_map
                        else:
                            condition_pose_str = b64decode(condition_image_item.get("pose_str", condition_image_item.get("binary", condition_image_item.get("b64_binary", condition_image_item.get("image","")))))
                            condition_pose_pil = Image.open(io.BytesIO(condition_pose_str)).convert("RGB")
                            if self.pose_transform is not None:
                                condition_pose = self.pose_transform(condition_pose_pil)
                                # condition_pose = condition_pose.float() / 255.0
                            res["src_pose_map"] = condition_pose
                        
                        # pdb.set_trace()
                        # TODO here, video dataloader
                        image_list = []
                        pose_map_list = []
                        mask_list = []

                        image_range = len(list(img_keys)) -1 # How to save memory and load more frames?
                        for i in range(image_range):
                            # print("Read {} image out of {} images".format(i+1,len(list(img_keys))))
                            image_item = data_item[list(img_keys)[i+1]]
                            image_str = b64decode(image_item.get("img_str", image_item.get("binary", image_item.get("b64_binary", image_item.get("image","")))))
                            image_pil = Image.open(io.BytesIO(image_str)).convert("RGB")
                            mask_str = b64decode(image_item.get("mask_str", image_item.get("binary", image_item.get("b64_binary", image_item.get("image","")))))
                            mask_pil = Image.open(io.BytesIO(mask_str)).convert("RGB")



                            if is_monochromatic_image(image_pil):
                                continue
                            if self.transform is not None:
                                image = self.transform(image_pil)
                            if image.std() < 0.02:
                                continue

                            if self.transform is not None:
                                mask = self.transform(mask_pil)
                            
                            if not self.v4:
                                if self.filter_pose:
                                    assert "openpose" in image_item
                                    candidate = np.array(image_item["openpose"]["bodies"]["candidate"])
                                    subset = np.array(image_item["openpose"]["bodies"]["subset"])
                                    if not filter_keypoints(image.shape[:2], candidate, subset):
                                        continue

                            image_list.append(image)
                            mask_list.append(mask)
                            H, W = image.shape[1:]

                            if not self.v4:
                                if self.with_pose:
                                    if "body+hand+face" in self.control_type: 
                                        pose = image_item["openpose"]
                                        pose_map = draw_pose(pose,H,W)
                
                                        pose_map = torch.from_numpy(pose_map.copy()).float() / 255.0
                                        pose_map = pose_map.permute(2,0,1)
                                        pose_map_list.append(pose_map)
                                    elif "body+face" in self.control_type:
                                        pose = image_item["openpose"]
                                        pose_map = draw_pose(pose,H,W,draw_hand=False)
                                        pose_map = torch.from_numpy(pose_map.copy()).float() / 255.0
                                        pose_map = pose_map.permute(2,0,1)
                                        pose_map_list.append(pose_map)
                                    elif "body" in self.control_type:
                                        pose = image_item["openpose"]
                                        pose_map = draw_pose(pose,H,W,draw_hand=False,draw_face=False)
                                        pose_map = torch.from_numpy(pose_map.copy()).float() / 255.0
                                        pose_map = pose_map.permute(2,0,1)
                                        pose_map_list.append(pose_map)
                            else:
                                pose_str = b64decode(image_item.get("pose_str", image_item.get("binary", image_item.get("b64_binary", image_item.get("image","")))))
                                pose_pil = Image.open(io.BytesIO(pose_str)).convert("RGB")
                                if self.pose_transform is not None:
                                    pose_map = self.pose_transform(pose_pil)
                                    # pose_map = pose_map.float() / 255.0
                                pose_map_list.append(pose_map)
                        res['image_list'] = torch.stack(image_list)
                        res['pose_map_list'] = torch.stack(pose_map_list)
                        res['mask_list'] = torch.stack(mask_list)
                        yield res

                    except Exception as e:
                        # if self.debug:
                        raise(e)
                        # else:
                        #     print("Dataset Exception:", e)
                else:
                    try:                    
                        data_item = json.loads(example)
                        img_keys = data_item.keys()
                        if len(img_keys)-self.frame_num-1 < 1:
                            continue
                        condition_int = 0
                        condition_image_item = data_item[list(img_keys)[condition_int]]
                        condition_image_str = b64decode(condition_image_item.get("img_str", condition_image_item.get("binary", condition_image_item.get("b64_binary", condition_image_item.get("image","")))))
                        condition_image_pil = Image.open(io.BytesIO(condition_image_str)).convert("RGB")
                        condition_mask_str = b64decode(condition_image_item.get("mask_str", condition_image_item.get("binary", condition_image_item.get("b64_binary", condition_image_item.get("image","")))))
                        condition_mask_pil = Image.open(io.BytesIO(condition_mask_str)).convert("RGB")
                        if is_monochromatic_image(condition_image_pil):
                            continue
                        if self.transform is not None:
                            condition_image = self.transform(condition_image_pil)
                        if condition_image.std() < 0.02:
                            continue
    
                        if self.transform is not None:
                            condition_mask = self.transform(condition_mask_pil)

                        if "text_blip" in condition_image_item:
                            text_blip = condition_image_item["text_blip"]
                        else:
                            pass
                        
                        if not self.v4:
                            if "text_bg" in condition_image_item:
                                text_bg = condition_image_item["text_bg"]
                            else:
                                pass

                            if len(text_bg) > 0 and text_bg[0] == ',': 
                                text_bg = text_bg[1:]
                            if isnumeric(text_bg):
                                continue
                            if porn_filter(text_bg):
                                continue
                            if not is_english(text_bg):
                                continue
                            # Load text from blip-v2
                            if len(text_blip) > 0 and text_blip[0] == ',': 
                                text_blip = text_blip[1:]
                            if text_blip==None or len(text_blip) == 0:
                                print('empty text_blip input: %s, will skip' % text_blip)
                                continue
                            if isnumeric(text_blip):
                                continue
                            if porn_filter(text_blip):
                                continue
                            if not is_english(text_blip):
                                continue

                            # res = {'condition_image': condition_image, 'face_mask': head_tensor, 'text_blip': text_blip, 'text_bg': text_bg, 'condition_mask': condition_mask}
                            res = {'condition_image': condition_image, 'text_blip': text_blip, 'text_bg': text_bg, 'condition_mask': condition_mask}
                        else:
                            res = {'condition_image': condition_image, 'condition_mask': condition_mask}

                        if self.mask_mode is not None and self.random_mask:
                            randommask = get_mask(mask_mode=self.mask_mode, img_size=condition_image.shape[1:])
                            res["randommask"] = randommask
                            
                        H, W = condition_image.shape[1:]

                        if not self.v4:
                            if self.with_pose:
                                if "body+hand+face" in self.control_type: 
                        
                                    src_pose = condition_image_item["openpose"]
                                    src_pose_map = draw_pose(src_pose,H,W)
            
                                    src_pose_map = torch.from_numpy(src_pose_map.copy()).float() / 255.0
                                    src_pose_map = src_pose_map.permute(2,0,1)
                                    res["src_pose_map"] = src_pose_map

                                elif "body+face" in self.control_type:

                                    src_pose = condition_image_item["openpose"]
                                    src_pose_map = draw_pose(src_pose,H,W,draw_hand=False)
                                    src_pose_map = torch.from_numpy(src_pose_map.copy()).float() / 255.0
                                    src_pose_map = src_pose_map.permute(2,0,1)
                                    res["src_pose_map"] = src_pose_map

                                elif "body" in self.control_type:

                                    src_pose = condition_image_item["openpose"]
                                    src_pose_map = draw_pose(src_pose,H,W,draw_hand=False,draw_face=False)
                                    src_pose_map = torch.from_numpy(src_pose_map.copy()).float() / 255.0
                                    src_pose_map = src_pose_map.permute(2,0,1)
                                    res["src_pose_map"] = src_pose_map
                        else:
                            # print(condition_image_item.keys())
                            condition_pose_str = b64decode(condition_image_item.get("pose_str", condition_image_item.get("binary", condition_image_item.get("b64_binary", condition_image_item.get("image","")))))
                            condition_pose_pil = Image.open(io.BytesIO(condition_pose_str)).convert("RGB")
                            if self.pose_transform is not None:
                                condition_pose = self.pose_transform(condition_pose_pil)
                                # condition_pose = condition_pose.float() / 255.0
                            res["src_pose_map"] = condition_pose

                        
                        # pdb.set_trace()
                        # TODO here, video dataloader
                        image_list = []
                        pose_map_list = []
                        mask_list = []

                        image_range = self.frame_num # How to save memory and load more frames?
                        for i in range(image_range):
                            
                            image_item = data_item[list(img_keys)[i+1]]

                            image_str = b64decode(image_item.get("img_str", image_item.get("binary", image_item.get("b64_binary", image_item.get("image","")))))
                            image_pil = Image.open(io.BytesIO(image_str)).convert("RGB")
                            mask_str = b64decode(image_item.get("mask_str", image_item.get("binary", image_item.get("b64_binary", image_item.get("image","")))))
                            mask_pil = Image.open(io.BytesIO(mask_str)).convert("RGB")



                            if is_monochromatic_image(image_pil):
                                continue
                            if self.transform is not None:
                                image = self.transform(image_pil)
                            if image.std() < 0.02:
                                continue

                            if self.transform is not None:
                                mask = self.transform(mask_pil)
                            
                            if not self.v4:
                                if self.filter_pose:
                                    assert "openpose" in image_item
                                    candidate = np.array(image_item["openpose"]["bodies"]["candidate"])
                                    subset = np.array(image_item["openpose"]["bodies"]["subset"])
                                    if not filter_keypoints(image.shape[:2], candidate, subset):
                                        continue

                            image_list.append(image)
                            mask_list.append(mask)
                            H, W = image.shape[1:]

                            if not self.v4:
                                if self.with_pose:
                                    if "body+hand+face" in self.control_type: 
                                        pose = image_item["openpose"]
                                        pose_map = draw_pose(pose,H,W)
                
                                        pose_map = torch.from_numpy(pose_map.copy()).float() / 255.0
                                        pose_map = pose_map.permute(2,0,1)
                                        pose_map_list.append(pose_map)
                                    elif "body+face" in self.control_type:
                                        pose = image_item["openpose"]
                                        pose_map = draw_pose(pose,H,W,draw_hand=False)
                                        pose_map = torch.from_numpy(pose_map.copy()).float() / 255.0
                                        pose_map = pose_map.permute(2,0,1)
                                        pose_map_list.append(pose_map)
                                    elif "body" in self.control_type:
                                        pose = image_item["openpose"]
                                        pose_map = draw_pose(pose,H,W,draw_hand=False,draw_face=False)
                                        pose_map = torch.from_numpy(pose_map.copy()).float() / 255.0
                                        pose_map = pose_map.permute(2,0,1)
                                        pose_map_list.append(pose_map)
                            else:
                                pose_str = b64decode(image_item.get("pose_str", image_item.get("binary", image_item.get("b64_binary", image_item.get("image","")))))
                                pose_pil = Image.open(io.BytesIO(pose_str)).convert("RGB")
                                if self.pose_transform is not None:
                                    pose_map = self.pose_transform(pose_pil)
                                    # pose_map = pose_map.float() / 255.0
                                pose_map_list.append(pose_map)
                        res['image_list'] = torch.stack(image_list)
                        res['pose_map_list'] = torch.stack(pose_map_list)
                        res['mask_list'] = torch.stack(mask_list)
                        yield res

                    except Exception as e:
                        # if self.debug:
                        raise(e)
                        # else:
                        #     print("Dataset Exception:", e)


def init_ds(train_data_path, *args, **kwargs):
    if isinstance(train_data_path, str):            
        train_data_path = [train_data_path]
    return ImageTextControlDataset(train_data_path, *args, **kwargs)


def tiktok_video_mm(**kwargs):
    if kwargs['v4']:
        train_data_path = "hdfs://harunava/home/byte_ailab_us_cvg/user/di.chang/dataset/TikTok-v4-train"
    else:
        train_data_path = "hdfs://harunava/home/byte_ailab_us_cvg/user/di.chang/dataset/TikTok-v3-train"
    return init_ds(train_data_path, with_pose=True, **kwargs)


def tiktok_video_mm_val(**kwargs):
    if kwargs['v4']:
        train_data_path = "/home/ICT2000/jefu/TikTok-v4/val_set" #"hdfs://harunava/home/byte_ailab_us_cvg/user/di.chang/dataset/TikTok-v4-test"
    else:
        train_data_path = "/home/ICT2000/jefu/TikTok-v4/val_set" #"hdfs://harunava/home/byte_ailab_us_cvg/user/di.chang/dataset/TikTok-v3-test"
    return init_ds(train_data_path, with_pose=True, **kwargs)


# from torchvision import transforms as T
# from dataset.transforms import RemoveWhite, CenterCrop
# image_transform = T.Compose([
#         #RemoveWhite(),
#         T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
#         T.RandomResizedCrop(
#                 64*8,
#                 scale=(1.0, 1.0), ratio=(1., 1.),
#                 interpolation=transforms.InterpolationMode.BILINEAR),
#         T.ToTensor(), 
#         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
# # image_transform = T.Compose([
# #             T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
# #             CenterCrop(),
# #             T.Resize((args.image_size*8, args.image_size*8)),
# #             T.ToTensor(), 
# #             T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# #         ])
# pose_transform = T.Compose([
#         T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
#         T.RandomResizedCrop(
#                 64*8,
#                 scale=(0.9, 1.0), ratio=(1., 1.),
#                 interpolation=T.InterpolationMode.BILINEAR),
#         T.ToTensor(), 
#     ])
# dataset = tiktok_video_mm_val(rank=0, transform=image_transform, control_type=["body+hand+face"], world_size=1, frame_num=16, train=True, v4=True,pose_transform=pose_transform)
# dataset.debug = True
# batch_size = 1
# loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1)
# loader_ = iter(loader)
# path = '/mnt/bn/dichang-bytenas/TikTok/sample_image_1'
# os.makedirs(path,exist_ok=True)
# def save_img(d, name="temp.jpg"):
#     import imageio
#     img = (d[0].permute(1,2,0).cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
#     imageio.imwrite(name, img)


# for i in range(10):
#     data = next(loader_)
#     image_list = data['image_list']
#     pose_map_list = data['pose_map_list']
#     mask_list = data['mask_list']
#     condition_image = data['condition_image']
#     condition_mask = data['condition_mask']
#     for img_num, image in enumerate(image_list):
#         os.makedirs(path+'/{}/'.format(i),exist_ok=True)
#         save_img(condition_image,name=path+'/{}/condition_image.jpg'.format(i))
#         save_img(condition_mask,name=path+'/{}/condition_mask.jpg'.format(i))
#         save_img(image,name=path+'/{}/image_{}.jpg'.format(i,img_num))
#         save_img(pose_map_list[img_num],name=path+'/{}/pose_{}.jpg'.format(i,img_num))
#         save_img(mask_list[img_num],name=path+'/{}/mask_{}.jpg'.format(i,img_num))