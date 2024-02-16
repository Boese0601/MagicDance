#!/usr/bin/python
# -*- coding: utf-8 -*-

import euler
import os
import io
import thriftpy2
import numpy as np

class FaceKeypointDetector(object):
    def __init__(self):
        module_path = os.path.dirname(__file__)
        self.common_thrift = thriftpy2.load(
            module_path + '/idls/common.thrift', 
            module_name='common_thrift'
        )
        self.server_thrift = thriftpy2.load(
            module_path + '/idls/faceattr.thrift', 
            module_name='faceattr_thrift'
        )
        self.client = euler.Client(
            self.server_thrift.FaceAttr,
            'sd://labcv.smash.serving?cluster=default',
            timeout=10
        )

    def predict(self, img, pt='106', return_bbox=False):
        # pt: 106->正常人脸关键点, 44->外轮廓关键点
        # https://bytedance.feishu.cn/wiki/wikcn1Y8WcbOFBHfVjNj0e6LVId
        # https://bytedance.feishu.cn/docs/doccnvCM1LYPhsMPfNRmZwQ371d#1LwosE
        if pt not in ['106', '44']:
            raise ValueError('%s keypoint type not supported yet.' % pt)

        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_data = buffer.getvalue()

        req = self.server_thrift.FaceAttrReq()
        base_info = self.common_thrift.BaseInfo()
        base_info.image_name = 'data'
        base_info.image_data = img_data
        req.base_info = base_info
        rsp = self.client.PredictFaceAttr(req)
        attr_infos = rsp.face_attr_infos
        
        h = rsp.base_meta.image_height
        w = rsp.base_meta.image_width

        keypoints = []
        bboxes = []
        if pt == '106':
            for attr_info in attr_infos:
                bbox = [attr_info.rect.left * w, attr_info.rect.top * h, attr_info.rect.right * w, attr_info.rect.bottom * h]
                bboxes.append(np.array(bbox))
                cur = [[p.x * w, p.y * h] for p in attr_info.landmark106s]
                keypoints.append(np.array(cur, dtype=np.float32).reshape((-1, 2)))
        else:
            for attr_info in attr_infos:
                bbox = [attr_info.rect.left * w, attr_info.rect.top * h, attr_info.rect.right * w, attr_info.rect.bottom * h]
                bboxes.append(np.array(bbox))
                cur = [[p.x * w, p.y * h] for p in attr_info.landmark_outlines][:44]
                keypoints.append(np.array(cur, dtype=np.float32).reshape((-1, 2)))
        if return_bbox:
            return keypoints, bboxes
        else:
            return keypoints
