import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
from . import util
from .body import Body
from .hand import Hand

body_estimation = None # Body('./model_lib/ControlNet/annotator/ckpts/body_pose_model.pth')
hand_estimation = None # Hand('./model_lib/ControlNet/annotator/ckpts/hand_pose_model.pth')


def apply_openpose(oriImg, hand=False, draw=True):
    global body_estimation, hand_estimation
    if body_estimation is None:
        body_estimation = Body('./model_lib/ControlNet/annotator/ckpts/body_pose_model.pth')
    oriImg = oriImg[:, :, ::-1].copy()
    with torch.no_grad():
        candidate, subset = body_estimation(oriImg)
        canvas = np.zeros_like(oriImg)
        if draw:
            canvas = util.draw_bodypose(canvas, candidate, subset)
        if hand:
            if hand_estimation is None:
                hand_estimation = Hand('./model_lib/ControlNet/annotator/ckpts/hand_pose_model.pth')
            hands_list = util.handDetect(candidate, subset, oriImg)
            all_hand_peaks = []
            for x, y, w, is_left in hands_list:
                peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
                peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
                peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
                all_hand_peaks.append(peaks)
            if draw:
                canvas = util.draw_handpose(canvas, all_hand_peaks)
        return canvas, dict(candidate=candidate.tolist(), subset=subset.tolist())
