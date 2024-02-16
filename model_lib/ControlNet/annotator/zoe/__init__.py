# ZoeDepth
# https://github.com/isl-org/ZoeDepth

import os
import cv2
import numpy as np
import torch

from einops import rearrange
from .zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth
from .zoedepth.utils.config import get_config
from model_lib.ControlNet.annotator.util import annotator_ckpts_path
import pdb

class ZoeDetector:
    def __init__(self):
        # remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/ZoeD_M12_N.pt"
        modelpath = os.path.join(annotator_ckpts_path, "ZoeD_M12_N.pt")
        if not os.path.exists(modelpath):
            raise FileNotFoundError
        conf = get_config("zoedepth", "infer")
        model = ZoeDepth.build_from_config(conf)
        model.load_state_dict(torch.load(modelpath)['model'])
        model = model.cuda()
        model.device = 'cuda'
        model.eval()
        self.model = model

    def __call__(self, input_image):
        assert input_image.ndim == 3
        image_depth = input_image
        with torch.no_grad():
            image_depth = image_depth / 255.0
            print(image_depth.shape)
            image_depth = image_depth.unsqueeze(0)
            # image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
            print(image_depth.shape)
            depth = self.model.infer(image_depth)
            print(depth.shape)
            depth = depth[0, 0].cpu().numpy()

            vmin = np.percentile(depth, 2)
            vmax = np.percentile(depth, 85)

            depth -= vmin
            depth /= vmax - vmin
            depth = 1.0 - depth
            depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)

            return depth_image
