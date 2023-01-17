import os
import datetime
import time
import copy
import cv2
import numpy as np
from logging import Logger
from typing import Tuple

import torch
from torchvision import transforms

from models.deepfake_detect.DeepfakeDetect import DeepfakeDetector, get_extracter, custom_data_process
from utils.utils import AverageMeter

import albumentations as A
from albumentations import Compose, OneOf


class UnNormalize:
    #restore from T.Normalize
    #反归一化
    def __init__(self,mean=(0.485, 0.456, 0.406),std= (0.229, 0.224, 0.225),device=None):
        self.mean=torch.tensor(mean).view((1,-1,1,1)).to(device)
        self.std=torch.tensor(std).view((1,-1,1,1)).to(device)
    def __call__(self,x):
        x=(x*self.std)+self.mean
        return torch.clip(x,0,None)


class DeepfakeEvaluate():
    """
    Process deepfake detection with and without defense module.
    Accepts video file.
    """
    def __init__(self, model, defenses:list, logger:Logger, batch_size=8, device=None, view_path=None) -> None:
        """
        Args:
            defense (model): Any callable defense model
            logger (Logger): Logger
        """
        self.logger = logger
        self.device = device if device is not None else torch.device('cuda')
        self.detector = DeepfakeDetector(model, self.device)
        self.extracter = get_extracter()
        self.defenses = defenses
        self.batch_size = batch_size
        self.view_path = view_path
        if self.view_path is not None:
            os.makedirs(self.view_path)
        self.save_count = 0
    
    def save_image(self, img, path):
        toPIL = transforms.ToPILImage()
        img = toPIL(img)
        img.save(path)

    def __call__(self, video_path:str, before_dfs=False, sampling_interval=8) -> Tuple[float, float]:
        """Run deepfake detect with defense for single video file.

        Args:
            `video_path` (str): path of the video file to be detected
            `before_dfs` (bool, optional): If `True`, detection will run both before and with defense
            , otherwise with defense only. Defaults to `False`.

        Returns:
            Tuple[float, float]: defended result (float), before defense result (float, -1 if before_dfs is False)
        """
        reader = cv2.VideoCapture(video_path)  # frames
        # fps = reader.get(cv2.CAP_PROP_FPS)
        
        prediction = AverageMeter()
        prediction_defense = AverageMeter()
        starttime = datetime.datetime.now()
        skip = False
        unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], self.device)

        while reader.isOpened():
            batch = []
            # t1 = time.time()
            for _ in range(self.batch_size):
                success, image = reader.read()  # image即单张图片
                if not success:
                    break
                for _ in range(sampling_interval-1):
                    success, _ = reader.read()  # image即单张图片
                    if not success:
                        break
                result = self.extracter.inference_batch(np.expand_dims(image, 0),scale=1.7)
                faces = result[0][0]
                if len(faces) > 1:
                    success = False
                    skip = True
                    break
                elif len(faces) == 0:
                    continue
                else:
                    batch.append(faces[0])
                if not success:
                    break
            # t2 = time.time()
            if len(batch) > 0:
                # before defense
                _batch_size = len(batch)
                if before_dfs:
                    batch_face, _, _ = custom_data_process(batch,img_label=0,size=380)  # 有bug，batch_face并不都是1
                    output = self.detector(batch_face)
                    output = output[:,1].mean().item()
                    prediction.append(output, count=_batch_size)
                # defense
                # faces_defense = copy.copy(batch)
                # for defense in self.defenses:
                #     faces_defense = [defense(face[None,:])[0] for face in faces_defense]
                # batch_face_defense, _, _ = custom_data_process(faces_defense, img_label=0, size=380)  # 有bug，batch_face并不都是1
                batch_face_defense, _, _ = custom_data_process(batch,img_label=0,size=380)
                if self.view_path is not None and self.save_count <= 3*100:
                    if self.save_count%3 == 0:
                        self.save_image(unnorm(batch_face_defense[0].to(self.device))[0], os.path.join(self.view_path, '{:0>3d}_before.png'.format(self.save_count//3)))
                # t3 = time.time()
                for defense in self.defenses:
                    batch_face_defense = defense(batch_face_defense)
                # t4 = time.time()
                # save samples
                if self.view_path is not None and self.save_count <= 3*100:
                    if self.save_count%3 == 0:
                        self.save_image(unnorm(batch_face_defense[0])[0], os.path.join(self.view_path, '{:0>3d}.png'.format(self.save_count//3)))
                        # self.save_image(batch_face_defense[0]-batch_face[0], os.path.join(self.view_path, '{:0>3d}_res.png'.format(self.save_count//3)))
                    self.save_count += 1
                output_defense = self.detector(batch_face_defense)
                output_defense = output_defense[:,1].mean().item()
                prediction_defense.append(output_defense, count=_batch_size)
                # t5 = time.time()

            # self.logger.info('read and extractor:{:.4f} defense:{:.4f} data_process:{:.5f} detector:{:.4f}'.format(t2-t1,t3-t2,t4-t3,t5-t4))
            
            if not success:
                break
            

        # 统计结果
        result_ori = prediction() if before_dfs else -1
        result_dfs = prediction_defense()
        
        result_msg = video_path
        if before_dfs:
            result_msg += " Before: {:.5f}".format(result_ori)
        result_msg += " After:{:.5f}".format(result_dfs)
        endtime = datetime.datetime.now()  # 结束时间
        spenttime = endtime-starttime
        result_msg += " Time: {}".format(str(spenttime)[:-5])
        self.logger.info(result_msg)
        
        return result_dfs, result_ori


class DeepfakeEvaluateTTA():
    def __init__(self, model, defenses:list, logger:Logger, batch_size=8, device=None, view_path=None) -> None:

        self.logger = logger
        self.detector = DeepfakeDetector(model, device)
        self.extracter = get_extracter()
        self.defenses = defenses
        self.batch_size = batch_size
        self.view_path = view_path
        if self.view_path is not None:
            os.makedirs(self.view_path)
        self.save_count = 0
    
    def save_image(self, img, path):
        toPIL = transforms.ToPILImage()
        img = toPIL(img)
        img.save(path)

    
    def tta(self, input):
        transform = Compose([
            A.RandomRotate90(),
            A.Flip(),
            # A.Transpose(),
            OneOf([
                A.GaussNoise(),
            ], p=0.3),
            OneOf([
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=.1),
                A.Blur(blur_limit=3, p=.1),
            ], p=0.5),
            # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
            OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
                A.PiecewiseAffine(p=0.3),
            ], p=0.5),
            OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
        ])
        return [transform(image=input)['image'] for _ in range(5)]



    def __call__(self, video_path:str, before_dfs=False, sampling_interval=8) -> Tuple[float, float]:
        reader = cv2.VideoCapture(video_path)  # frames
        # fps = reader.get(cv2.CAP_PROP_FPS)
        
        prediction = AverageMeter()
        prediction_defense = AverageMeter()
        starttime = datetime.datetime.now()
        skip = False

        while reader.isOpened():
            batch = []
            # t1 = time.time()
            for _ in range(self.batch_size):
                success, image = reader.read()  # image即单张图片
                if not success:
                    break
                for _ in range(sampling_interval-1):
                    success, _ = reader.read()  # image即单张图片
                    if not success:
                        break
                result = self.extracter.inference_batch(np.expand_dims(image, 0),scale=1.7)
                faces = result[0][0]
                if len(faces) > 1:
                    success = False
                    skip = True
                    break
                elif len(faces) == 0:
                    continue
                else:
                    batch+=self.tta(faces[0])
                if not success:
                    break
            # t2 = time.time()
            if len(batch) > 0:
                # before defense
                _batch_size = len(batch)
                if before_dfs:
                    batch_face, _, _ = custom_data_process(batch,img_label=0,size=380)  # 有bug，batch_face并不都是1
                    output = self.detector(batch_face)
                    output = output[:,1].mean().item()
                    prediction.append(output, count=_batch_size)
                batch_face_defense, _, _ = custom_data_process(batch,img_label=0,size=380)
                # t3 = time.time()
                # for defense in self.defenses:
                #     batch_face_defense = defense(batch_face_defense)
                # t4 = time.time()
                # save samples
                if self.view_path is not None and self.save_count <= 3*100:
                    self.save_count += 1
                    if self.save_count%3 == 0:
                        self.save_image(batch_face_defense[0], os.path.join(self.view_path, '{:0>3d}.png'.format(self.save_count//3)))
                output_defense = self.detector(batch_face_defense)
                output_defense = output_defense[:,1].mean().item()
                prediction_defense.append(output_defense, count=_batch_size)
                # t5 = time.time()

            # self.logger.info('read and extractor:{:.4f} defense:{:.4f} data_process:{:.5f} detector:{:.4f}'.format(t2-t1,t3-t2,t4-t3,t5-t4))
            
            if not success:
                break
            

        # 统计结果
        result_ori = prediction() if before_dfs else -1
        result_dfs = prediction_defense()
        
        result_msg = video_path
        if before_dfs:
            result_msg += " Before: {:.5f}".format(result_ori)
        result_msg += " After:{:.5f}".format(result_dfs)
        endtime = datetime.datetime.now()  # 结束时间
        spenttime = endtime-starttime
        result_msg += " Time: {}".format(str(spenttime)[:-5])
        self.logger.info(result_msg)
        
        return result_dfs, result_ori

