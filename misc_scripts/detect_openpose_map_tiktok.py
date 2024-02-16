import sys
import argparse
import numpy as np 
import cv2
import json 
from utils.utils import split as split_list
from utils.utils import chunk as chunk_iterable

from dataloader import KVWriter, KVReader
from tqdm import tqdm 
import json
from base64 import b64decode, b64encode
from PIL import Image
import io, os
from typing import IO, Any, List
import torch
from multiprocessing import Pool, current_process
import multiprocessing
import torch.nn.functional as F
import torchvision.transforms as T
from dataset.hdfs_io import hopen, hlist_files, hexists, hcopy
import time
from misc_scripts.keypoint import FaceKeypointDetector

from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch


import numpy as np 
import cv2
import json 
from copy import deepcopy 

from model_lib.ControlNet.annotator.openpose.body import Body
from model_lib.ControlNet.annotator.openpose.hand import Hand
from model_lib.ControlNet.annotator.openpose.face import Face
# from model_lib.ControlNet.annotator.openpose import apply_openpose
# from misc_scripts.face_detect.service_caller import check_face_from_cv2_im
from dataset.transforms import RemoveWhite, CenterCrop
from multiprocessing import Pool
import pdb
from model_lib.ControlNet.annotator.openpose import util

        
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

def pil2cv2(im_pil):
    return cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)

def cv22str(im_cv2):
    im_pil = Image.fromarray(cv2.cvtColor(im_cv2, cv2.COLOR_BGR2RGB))

    io_bytes = io.BytesIO()
    im_pil.save(io_bytes, 'jpeg')
    img_str = b64encode(io_bytes.getvalue()).decode('ascii')
    return img_str


def filter_keypoints(image_size, candidate, subset, max_people=1):
    if len(subset) > max_people:
        return False
    for n in range(len(subset)):
        count = 0
        min_x = min_y = 999999
        max_x = max_y = 0
        for i in range(18):
            index = int(subset[n][i])
            if index == -1:
                continue
            count += 1
            x, y = candidate[index][0:2]
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x), max(max_y, y)
        
        image_area = image_size[0] * image_size[1]
        subject_area = max(0, max_x-min_x) * max(0, max_y-min_y)
        area_ratio = subject_area / image_area

        if count > 6 and area_ratio > 0.04:
            return True
    return False

def detect_pose(input_folder, output_folder):
    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)
        if os.path.isdir(folder_path):
            output_folder_path = os.path.join(output_folder, folder_name)
            os.makedirs(output_folder_path, exist_ok=True)
            for image_name in os.listdir(folder_path):
                if image_name.endswith('.jpg') or image_name.endswith('.png'):
                    image_path = os.path.join(folder_path, image_name)
                    output_image_path = os.path.join(output_folder_path, image_name)
                    worker = Worker()
                    new_values = worker(image_path)

                    if new_values is not None:
                        src_pose = new_values['openpose']
                        pose_map_draw = draw_pose(src_pose, new_values['H'], new_values['W'])
                        pose_map = Image.fromarray(pose_map_draw)
                        print("Save Pose Map to {}".format(output_image_path))
                        pose_map.save(output_image_path)
                    else:
                        print("Invalid pose for {}".format(image_path))
                        continue



class Worker:
    body_model = None # initialize CUDA modules only inside subprocesses 
    hand_model = None
    face_model = None
    # text_model = None
    def __init__(self, body_ckpt_path="./model_lib/ControlNet/annotator/ckpts/body_pose_model.pth",hand_ckpt_path='./model_lib/ControlNet/annotator/ckpts/hand_pose_model.pth', face_ckpt_path='./model_lib/ControlNet/annotator/ckpts/facenet.pth', image_size=(512,512), local_rank=0):
        # object variables will be copied to subprocesses every time worker is called
        if not os.path.exists(body_ckpt_path):
            if local_rank == 0:
                os.system(f"hdfs dfs -get hdfs://harunava/home/byte_ailab_us_cvg/user/di.chang/ControlNet/pretrained_weights/body_pose_model.pth {body_ckpt_path}")
            else:
                while not os.path.exists(body_ckpt_path):
                    time.sleep(1)
        if not os.path.exists(hand_ckpt_path):
            if local_rank == 0:
                os.system(f"hdfs dfs -get hdfs://harunava/home/byte_ailab_us_cvg/user/di.chang/ControlNet/pretrained_weights/body_pose_model.pth {hand_ckpt_path}")
            else:
                while not os.path.exists(hand_ckpt_path):
                    time.sleep(1)
        if not os.path.exists(face_ckpt_path):
            if local_rank == 0:
                os.system(f"hdfs dfs -get hdfs://harunava/home/byte_ailab_us_cvg/user/di.chang/ControlNet/pretrained_weights/body_pose_model.pth {face_ckpt_path}")
            else:
                while not os.path.exists(face_ckpt_path):
                    time.sleep(1)
        self.body_ckpt_path = body_ckpt_path
        self.hand_ckpt_path = hand_ckpt_path
        self.face_ckpt_path = face_ckpt_path
        self.image_size = image_size

    def __call__(self, image_path):
        process = current_process()
        # idx = process._identity[0]
        device = torch.device("cuda", 0)
        # device = torch.device("cuda", idx % torch.cuda.device_count())
        torch.cuda.set_device(device)
        if Worker.body_model is None or Worker.hand_model is None or Worker.face_model is None:
            # print(f"initializing {process.name} pid={process.pid} idx={idx}")
            Worker.body_model = Body(self.body_ckpt_path)
            Worker.body_model.model.to(device)
            Worker.hand_model = Hand(self.hand_ckpt_path)
            Worker.hand_model.model.to(device)
            Worker.face_model = Face(self.face_ckpt_path)
            Worker.face_model.model.to(device)
            # Worker.text_model = Blip2ForConditionalGeneration.from_pretrained(
            #     "./model_lib/ControlNet/annotator/ckpts/blip2-opt-2.7b", torch_dtype=torch.float16
            # )
            # Worker.text_model.to(device)
        body_estimation = Worker.body_model
        hand_estimation = Worker.hand_model
        face_estimation = Worker.face_model

        image_transform = T.Compose([
                    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                    # CenterCrop(),
                    # T.Resize(self.image_size),
                ])
        # new_values = {}
        # for image_path in [os.path.join(value, image) for image in os.listdir(value)]:
        #     image_id = image_path.split('/')[-1]
        #     mask_path = deepcopy(image_path)
        #     mask_path = mask_path.replace('cropped_images','masks')
            # image_pil = Image.open(image_path).convert("RGB")
            # image_pil = image_transform(image_pil)
        #     mask_pil = Image.open(mask_path).convert("RGB")
        #     mask_pil = image_transform(mask_pil)
            #### Hugging_face_BLIP2
        image_pil = Image.open(image_path).convert("RGB")
        image_pil = image_transform(image_pil)
            # processor = Blip2Processor.from_pretrained("./model_lib/ControlNet/annotator/ckpts/blip2-opt-2.7b")
            # inputs = processor(images=image_pil, return_tensors="pt").to(device, torch.float16)
            # prompt = "Question: What is the background in the image? Answer:"
            # inputs_bg = processor(images=image_pil, text=prompt, return_tensors="pt").to(device, torch.float16)
            # generated_ids_bg = Worker.text_model.generate(**inputs_bg)
            # generated_text_bg = processor.batch_decode(generated_ids_bg, skip_special_tokens=True)[0].strip()

            # generated_ids = Worker.text_model.generate(**inputs)
            # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            #### Hugging_face_BLIP2
            # print("text:",text)
            # print("BLIP:",generated_text)
        image = np.asarray(image_pil)
        try:
            new_image = image[:, :, ::-1].copy()
            H, W, C = image.shape
            # print(H,W)
            with torch.no_grad():
                candidate, subset = body_estimation(new_image)
                hands = []
                faces = []
                # Hand
                hands_list = util.handDetect(candidate, subset, new_image)
                for x, y, w, is_left in hands_list:
                    peaks = hand_estimation(new_image[y:y+w, x:x+w, :]).astype(np.float32)
                    if peaks.ndim == 2 and peaks.shape[1] == 2:
                        peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
                        peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
                        hands.append(peaks.tolist())
                # Face
                faces_list = util.faceDetect(candidate, subset, new_image)
                for x, y, w in faces_list:
                    heatmaps = face_estimation(new_image[y:y+w, x:x+w, :])
                    peaks = face_estimation.compute_peaks_from_heatmaps(heatmaps).astype(np.float32)
                    if peaks.ndim == 2 and peaks.shape[1] == 2:
                        peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
                        peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
                        faces.append(peaks.tolist())

            valid = filter_keypoints(image.shape[:2], candidate, subset)
        except:
            valid = False
            
            # print(valid)

        if valid:
            if candidate.ndim == 2 and candidate.shape[1] == 4:
                candidate = candidate[:, :2]
                candidate[:, 0] /= float(W)
                candidate[:, 1] /= float(H)
            bodies = dict(candidate=candidate.tolist(), subset=subset.tolist())
            new_values = {}

            io_bytes = io.BytesIO()
            mask_io_bytes = io.BytesIO()
            image_pil.save(io_bytes, 'jpeg')
            # img_str = b64encode(io_bytes.getvalue()).decode('ascii')
            # mask_pil.save(mask_io_bytes, 'jpeg')
            mask_str = b64encode(mask_io_bytes.getvalue()).decode('ascii')
            # new_values[image_id]['img_str'] = img_str
            # new_values[image_id]['mask_str'] = mask_str
            # new_values[image_id]['text_bg'] = generated_text_bg
            # new_values[image_id]['text_blip'] = generated_text
            new_values["openpose"] = dict(bodies=bodies, hands=hands, faces=faces)
            new_values["H"] = H
            new_values["W"] = W
            return new_values
        else:
            return None

# def detect_pose_function(image):
#     # This is a placeholder function, replace it with your actual detect_pose function
#     # It should take an image as input and return the detected pose map
#     # Example: 
#     # pose_map = your_detect_pose_function(image)
#     return image

# Define the paths
# input_folder = "train_set"
# output_folder = "pose_map_train_set"


# Call the function


# detect_pose(input_folder, output_folder)


import multiprocessing

def process_folder(folder_path, output_folder):
    if os.path.isdir(folder_path):
        folder_name = os.path.basename(folder_path)
        output_folder_path = os.path.join(output_folder, folder_name)
        os.makedirs(output_folder_path, exist_ok=True)

        for image_name in os.listdir(folder_path):
            if image_name.endswith('.jpg') or image_name.endswith('.png'):
                image_path = os.path.join(folder_path, image_name)
                output_image_path = os.path.join(output_folder_path, image_name)
                worker = Worker()
                new_values = worker(image_path)

                if new_values is not None:
                    src_pose = new_values['openpose']
                    pose_map_draw = draw_pose(src_pose, new_values['H'], new_values['W'])
                    pose_map = Image.fromarray(pose_map_draw)
                    print("Save Pose Map to {}".format(output_image_path))
                    pose_map.save(output_image_path)
                else:
                    print("Invalid pose for {}".format(image_path))

def process_folder_parallel(input_folder, output_folder, num_processes):
    pool = multiprocessing.Pool(processes=num_processes)

    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)
        pool.apply_async(process_folder, args=(folder_path, output_folder))

    pool.close()
    pool.join()

if __name__ == '__main__':
    num_processes = 12 
    input_folder = "/mnt/bn/dichang-usc/TikTok_data/TikTok-v4/disco_test_set"
    output_folder = "/mnt/bn/dichang-usc/TikTok_data/TikTok-v4/pose_map_disco_test_set"
    process_folder_parallel(input_folder, output_folder, num_processes)
