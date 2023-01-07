from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import matplotlib.pyplot as plt

from models.losses import FocalLoss
from external.nms import soft_nms
from models.decode import ctdet_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger
from .util import ModelOutputs, preprocess_image
from .base_detector_attack_release import BaseDetector

class CtdetDetector(BaseDetector):
    def __init__(self, opt):
        super(CtdetDetector, self).__init__(opt)

    def process(self, images, return_time=False):
        ################################################################################
        # Dense Category Attack
        max_0 = torch.max(images.data[:, 0, :, :])
        max_1 = torch.max(images.data[:, 1, :, :])
        max_2 = torch.max(images.data[:, 2, :, :])
        min_0 = torch.min(images.data[:, 0, :, :])
        min_1 = torch.min(images.data[:, 1, :, :])
        min_2 = torch.min(images.data[:, 2, :, :])
        max_pixel = torch.max(images.data.abs())
        eps = 0.05 * max_pixel
        iters = 3
        cates = 80  # 80 for COCO, 20 for PascalVOC
        attack_thres = 0.1
        eps_each_iter = eps/iters
        x_adv = torch.autograd.Variable(images.data, requires_grad=True)
        ur = images[0].cpu().numpy().shape[1:]
        crit = torch.nn.CrossEntropyLoss()
        hm_ori = self.model(images)[-1]['hm']
        hm_ori_sig = hm_ori.sigmoid()
        noise_tot = torch.zeros_like(x_adv.data)
        for iter in range(iters):#这个是PGD的循环
            x_adv_temp = torch.autograd.Variable(x_adv.data, requires_grad=True)
            hm_adv = self.model(x_adv_temp)[-1]['hm']
            hm_adv_sig = hm_adv.sigmoid()#将每个值套进sigmoid函数，将值映射在[0，1]
            #原值越高，映射出的值越高，值越高，越有可能是kp
            #因为只有将值映射到0，1，才能和attack_thres作比较

            noise = torch.zeros_like(x_adv_temp)
            #CWA中对类的循环
            for cate in range(cates):#这个是类的循环
                #确定目标点
                label_loca_adv = (hm_adv_sig[:, cate, :, :] > attack_thres).nonzero()
                #removepixel

                label_temp = torch.LongTensor([cate]).to(torch.device('cuda'))

                if len(label_loca_adv) == 0:
                    continue

                loss_count = 0
                #对每类中的目标点循环
                for index, item in enumerate(label_loca_adv):
                    #item中保存目标点的横纵坐标
                    if hm_ori_sig[:, cate, item[1], item[2]] > attack_thres:
                    #计算梯度
                        if loss_count == 0:
                            loss = crit(hm_adv[:, :, item[1], item[2]], label_temp)
                        else:
                            loss += crit(hm_adv[:, :, item[1], item[2]], label_temp)
                        loss_count += 1

                if loss_count == 0:
                    continue

                self.model.zero_grad()
                if x_adv_temp.grad is not None:
                    x_adv_temp.grad.data.fill_(0)

                loss.backward(retain_graph=True)
                noise_now = x_adv_temp.grad / x_adv_temp.grad.abs().max()
                noise += noise_now

            if iter == 0:
                grad_cam_10 = GradCam(model=self.model, feature_module=self.model.layer1, target_layer_names=["1"],
                                  use_cuda=torch.device('cuda'))
                mask10_ = torch.from_numpy(mask_Resize(grad_cam_10(images) > 0.5, ur)).to(torch.device('cuda'))
            noise_tot = noise_tot + (noise.data.sign().mul(eps_each_iter).mul(mask10_)).data
            noise_total_ = noise_tot[0].cpu().numpy()

            p2 = (np.linalg.norm(noise_total_[0]) + np.linalg.norm(noise_total_[1]) + np.linalg.norm(
                noise_total_[2])) / (512 * 512)
            count = np.count_nonzero(noise_total_)
            print('L2 norm of pertubation:{} '.format(p2))
            print('L0 norm of pertubation:{} '.format(count / (3 * 512 * 512)))

            x_adv.data += noise.data.sign().mul(eps_each_iter).mul(mask10_)
            x_adv.data[:, 0, :, :].clamp_(min_0, max_0)
            x_adv.data[:, 1, :, :].clamp_(min_1, max_1)
            x_adv.data[:, 2, :, :].clamp_(min_2, max_2)
            x_adv = torch.autograd.Variable(x_adv.data, requires_grad=True)

        output = self.model(x_adv)[-1]
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output['reg'] if self.opt.reg_offset else None

        if self.opt.flip_test:
            hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
            wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
            reg = reg[0:1] if reg is not None else None
        torch.cuda.synchronize()
        forward_time = time.time()
        dets,sc = ctdet_decode(hm, wh, reg=reg, K=self.opt.K)

        if return_time:
            return output, dets, forward_time, x_adv, noise_tot
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            if len(self.scales) > 1 or self.opt.nms:
                soft_nms(results[j], Nt=0.5, method=2)
        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def debug(self, debugger, images, dets, output, img_name, noise, scale=1):
        detection = dets.detach().cpu().numpy().copy()
        detection[:, :, :4] *= self.opt.down_ratio
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)
            pred = debugger.gen_colormap(
                output['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
            '''for k in range(len(dets[i])):
                if detection[i, k, 4] > self.opt.center_thresh:
                    debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                           detection[i, k, 4],
                                           img_id='out_pred_{:.1f}'.format(scale))'''

    def debug_noise(self, debugger, images, dets, output, img_name, noise, scale=1):
        detection = dets.detach().cpu().numpy().copy()
        detection[:, :, :4] *= self.opt.down_ratio
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)

            p_mean = np.array([0.485, 0.456, 0.406],
                              dtype=np.float32).reshape(1, 1, 3)
            p_std = np.array([0.229, 0.224, 0.225],
                             dtype=np.float32).reshape(1, 1, 3)

            noise = noise[i].detach().cpu().numpy().transpose(1, 2, 0)
            noise_ = ((noise * 30 * p_std + p_mean) * 255).astype(np.uint8)
            debugger.add_img(noise_, img_id='noise_img_{:.1f}'.format(scale))
            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))

    def show_results(self, debugger, image, results, img_name=''):
        debugger.add_img(image, img_id='ctdet')
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[4] > self.opt.vis_thresh:
                    debugger.add_coco_bbox(
                        bbox[:4], j - 1, bbox[4], img_id='ctdet')
        debugger.show_all_imgs(pause=self.pause)
        debugger.save_all_imgs(path='D:/python/object_detection/CenterNet/outputs/attack/', genID=True)


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)


    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())

        x_adv_temp = torch.autograd.Variable(input.data, requires_grad=True)
        attack_thres = 0.5
        hm_ori = output
        hm_ori_sig = hm_ori.sigmoid()
        crit = torch.nn.CrossEntropyLoss()
        hm_adv = output
        hm_adv_sig = hm_adv.sigmoid()  # 将每个值套进sigmoid函数，将值映射在[0，1]
            # 原值越高，映射出的值越高，值越高，越有可能是kp
            # 因为只有将值映射到0，1，才能和attack_thres作比较

        noise = torch.zeros_like(x_adv_temp)
        for cate in range(80):  # 这个是类的循环
                # 确定目标点
            label_loca_adv = (hm_adv_sig[:, cate, :, :] > attack_thres).nonzero()
                # removepixel

            label_temp = torch.LongTensor([cate]).to(torch.device('cuda'))

            if len(label_loca_adv) == 0:
                    continue
            loss_count = 0
                # 对每类中的目标点循环
            for index, item in enumerate(label_loca_adv):
                    # item中保存目标点的横纵坐标
                if hm_ori_sig[:, cate, item[1], item[2]] > attack_thres:
                        # 计算梯度
                    if loss_count == 0:
                        loss = crit(hm_adv[:, :, item[1], item[2]], label_temp)
                    else:
                        loss += crit(hm_adv[:, :, item[1], item[2]], label_temp)
                    loss_count += 1
            if loss_count == 0:
                continue
            self.model.zero_grad()
            if x_adv_temp.grad is not None:
                x_adv_temp.grad.data.fill_(0)
            # 每类反传一次梯度
        loss.backward(retain_graph=True)
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        print('grad list length {}'.format(len(grads_val)))
        print('grad size {}'.format(grads_val[-1].shape))
        #print('features size {}'.format(features[0].shape))

        # 如果feature_module=model.layer1 target_layer_names=0,1,2       梯度尺寸[1,256,56,56]
        # 如果feature_module=model.layer2 target_layer_names=0,1,2,3     梯度尺寸[1,512,28,28]
        # 如果feature_module=model.layer3 target_layer_names=0,1,2,3,4,5 梯度尺寸[1,1024,14,14]
        # 如果feature_module=model.layer4 target_layer_names=0,1,2       梯度尺寸[1,2048,7,7]
        # 代表着网络结构中每层layer中的bottleneck模块数 [3,4,6,3]

        target = features[-1]# feature是个列表，取最后一个 target size =  [1,256,56,56]
        target = target.cpu().data.numpy()[0, :]
        #print('target size {}'.format(target.shape))
        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        print('weights size {}'.format(weights.shape))
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        for i,w in enumerate(weights):
            # 循环的是channel数 将所有channel中的（梯度*系数）相加
            # 而weights来自于target_layer的梯度
            cam += w * target[i, :, :]  # w grad      target  参数

        #cam = np.maximum(cam, 0) # 去除负的元素
       # cam = cv2.resize(cam, input.shape[2:]) #直接resize至原始图片尺寸
        cam = cam - np.min(cam)  # 归一化
        cam = cam / np.max(cam)
        return cam

def mask_Resize(src, shape):
    height, width = src.shape
    dst_width, dst_height = shape
    if ((dst_height == height) and (dst_width == width)):
        return src
    dst_Image = np.zeros((dst_height, dst_width), np.uint8)
    scale_x = float(width) / dst_width
    scale_y = float(height) / dst_height
    for dst_y in range(dst_height):
        for dst_x in range(dst_width):
            src_x = (dst_x + 0.5) * scale_x - 0.5
            src_y = (dst_y + 0.5) * scale_y - 0.5
            src_x_0 = int(np.floor(src_x))
            src_y_0 = int(np.floor(src_y))
            src_x_1 = min(src_x_0 + 1, width - 1)
            src_y_1 = min(src_y_0 + 1, height - 1)
            value0 = (src_x_1 - src_x) * src[src_y_0, src_x_0] + (src_x - src_x_0) * src[src_y_0, src_x_1]
            value1 = (src_x_1 - src_x) * src[src_y_1, src_x_0] + (src_x - src_x_0) * src[src_y_1, src_x_1]
            dst_Image[dst_y, dst_x] = int((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1)
    return dst_Image