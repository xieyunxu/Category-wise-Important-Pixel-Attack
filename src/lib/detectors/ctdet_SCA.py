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
import sys

from models.losses import FocalLoss
from external.nms import soft_nms
#from models.decode import ctdet_decode, _nms, _max_pooling
from models.decode import ctdet_decode, _nms
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger

from .base_detector_attack_release import BaseDetector


class CtdetDetector(BaseDetector):
    def __init__(self, opt):
        super(CtdetDetector, self).__init__(opt)

    def process(self, images, return_time=False):

        ################################################################################
        # Sparse Category Attack

        paras = {}
        paras['eps'] = 0.05
        paras['iters_deepfool'] = 1
        paras['iters_sparse'] = 10
        paras['iters_sparse_cate'] = 10
        paras['vis_thresh'] = 0.3
        paras['attack_conf'] = 0.1
        paras['targets'] = 1
        paras['lambda'] = 0.02
        paras['momentum'] = 0
        paras['delta'] = 0.1
        paras['epsilon'] = 0.05

        images_source = copy.deepcopy(images)

        im_box = np.abs((np.ones((images.shape[2], images.shape[3], 3), dtype=np.float32) * (paras['epsilon'] * 3.4602)).astype(np.float32))
        im_box = im_box.transpose(2, 0, 1).reshape(1, 3, images.shape[2], images.shape[3])
        im_box = torch.from_numpy(im_box).to(self.opt.device)
        box_temp = copy.deepcopy(images)
        box_max = box_temp + im_box
        box_min = box_temp - im_box

        im_max = ((np.ones((images.shape[2], images.shape[3], 3), dtype=np.float32) - self.mean) / self.std).astype(np.float32)
        im_max = im_max.transpose(2, 0, 1).reshape(1, 3, images.shape[2], images.shape[3])
        im_max = torch.from_numpy(im_max).to(self.opt.device)
        im_min = ((np.zeros((images.shape[2], images.shape[3], 3), dtype=np.float32) - self.mean) / self.std).astype(np.float32)
        im_min = im_min.transpose(2, 0, 1).reshape(1, 3, images.shape[2], images.shape[3])
        im_min = torch.from_numpy(im_min).to(self.opt.device)

        box_max = self.clip_image_values(box_max, im_min, im_max)
        box_min = self.clip_image_values(box_min, im_min, im_max)

        x_i = copy.deepcopy(images)

        hm_ori = self.model(images)[-1]['hm']
        hm_ori_sig = hm_ori.sigmoid()

        main_label_loca_ori = (hm_ori_sig[:, :, :, :].max(dim=1)[0] > paras['vis_thresh']).nonzero()

        try:
            zeros_tensor = torch.zeros_like(main_label_loca_ori).sum(dim=1, keepdim=True)
            main_label_loca_ori_points = torch.cat((main_label_loca_ori.data, zeros_tensor), dim=1).float()
        except:
            main_label_loca_ori_points = main_label_loca_ori.data

        temp_argmax = hm_ori_sig[:, :, :, :].argmax(dim=1)[0]
        temp_max = hm_ori_sig[:, :, :, :].max(dim=1)[0][0]

        for index, item in enumerate(main_label_loca_ori):
            main_label_loca_ori[index, 0] = temp_argmax[item[1], item[2]]
            main_label_loca_ori_points[index, 3] = temp_max[item[1], item[2]]

        valid_cates = set()
        valid_cates_points = {}

        img_changed = 0.0

        if len(main_label_loca_ori) != 0:
            valid_cates = set(main_label_loca_ori[:, 0].to('cpu').numpy())

            mask_set = self.gen_mask(hm_ori_sig, valid_cates, paras)

            for iter in range(paras['iters_sparse']):

                main_hm_adv = self.model(x_i)[-1]['hm']
                main_hm_adv_sig = main_hm_adv.sigmoid().data

                main_hm_adv_only_max = main_hm_adv.sigmoid().data
                main_hm_adv_only_max = main_hm_adv_only_max / torch.max(main_hm_adv_only_max, dim=1, keepdim=True)[0]
                max_mask = (main_hm_adv_only_max == 1.0).float()

                empty_cate = 0
                attacking_cates = []
                for cate in valid_cates:
                    valid_cates_points[cate] = 0.0
                for key in valid_cates_points:
                    label_loca_adv_temp = (main_hm_adv_sig.mul(max_mask).mul(mask_set[key]) > paras['attack_conf']).nonzero()
                    if len(label_loca_adv_temp) != 0:
                        attacking_cates.append(key)
                        valid_cates_points[key] = main_hm_adv_sig[:, key, label_loca_adv_temp[:, 2], label_loca_adv_temp[:, 3]].sum()
                    else:
                        empty_cate += 1

                if empty_cate == len(valid_cates):
                    break

                temp_max = 0
                attack_cate = 0
                for name in valid_cates_points:
                    if valid_cates_points[name] > temp_max:
                        attack_cate = name
                        temp_max = valid_cates_points[name]

                label_loca_adv_temp = (main_hm_adv_sig[:, attack_cate, :, :] > paras['attack_conf']).nonzero()
                try:
                    label_loca_adv_temp[:, 0] = int(attack_cate)
                except:
                    print(1)

                normal_tot = torch.zeros_like(x_i)
                for iter_cate in range(paras['iters_sparse_cate']):

                    torch.cuda.empty_cache()

                    x_adv, r_tot, normal = self.deepfool_fast_onecate(copy.deepcopy(x_i.contiguous().data), paras, label_loca_adv_temp, valid_cates, lambda_=3)

                    normal_tot = paras['momentum']*normal_tot + normal.data

                    x_i = self.linear_solver(copy.deepcopy(x_i.data), normal, x_adv, box_max, box_min)

                    x_adv_test = images + (1+paras['lambda'])*(x_i-images)
                    onecate_hm_adv = self.model(x_adv_test)[-1]['hm']
                    onecate_hm_adv_sig = onecate_hm_adv.sigmoid().data

                    cate_attack_count = 0
                    for index, item in enumerate(label_loca_adv_temp):
                        if onecate_hm_adv_sig[:, :, item[1], item[2]].argmax() != item[0]:
                            cate_attack_count += 1

                    if cate_attack_count / len(label_loca_adv_temp) >= 0.99:
                        break

                main_hm_adv = self.model(x_i)[-1]['hm']
                main_hm_adv_sig = main_hm_adv.sigmoid().data

                temp_argmax = main_hm_adv_sig[:, :, :, :].argmax(dim=1)[0]
                temp_max = main_hm_adv_sig[:, :, :, :].max(dim=1)[0][0]

                main_attack_count = 0
                for index, item in enumerate(main_label_loca_ori):
                    if main_label_loca_ori[index, 0] == temp_argmax[item[1], item[2]]:
                        main_label_loca_ori_points[index, 3] = temp_max[item[1], item[2]]
                    else:
                        main_label_loca_ori_points[index, 3] = 0.0
                        main_attack_count += 1

                torch.cuda.empty_cache()

        r_tot = x_i - images
        #r_shape = r_tot.shape

        ################################################################################
        print("SCA")
        p_mean = np.array([0.485, 0.456, 0.406],
                          dtype=np.float32).reshape(3, 1, 1)
        p_std = np.array([0.229, 0.224, 0.225],
                         dtype=np.float32).reshape(3, 1, 1)
        noise_total_ = r_tot[0].cpu().numpy()
        r_shape = noise_total_[0].shape
        print('noise_total[0] shape: {}'.format(r_shape))
        p2 = (np.linalg.norm(noise_total_[0],ord=2) + np.linalg.norm(noise_total_[1],ord=2) + np.linalg.norm(noise_total_[2],ord=2)) \
             / (512 * 512)
        print('L2 norm of pertubation:{} '.format(p2))
        count = np.count_nonzero(noise_total_)
        print('L0 norm of pertubation:{} '.format(count / (3 * 512 * 512)))
        output = self.model(x_i)[-1]
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

        ################################################################################

        if return_time:
            return output, dets, forward_time, x_i, r_tot
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
            noise = ((noise * 100 * p_std + p_mean) * 255).astype(np.uint8)

            debugger.add_img(noise, img_id='noise_img_{:.1f}'.format(scale))

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

    def linear_solver(self, x_0, normal, boundary_point, max, min):

        input_shape = x_0.size()

        coord_vec = copy.deepcopy(normal)
        print('coord_vec size:',coord_vec.shape)
        #print('',coord_vec.shape)
        #print(coord_vec.view(-1).shape)
        plane_normal = copy.deepcopy(coord_vec).contiguous().view(-1)
        plane_point = copy.deepcopy(boundary_point).contiguous().view(-1)

        x_i = copy.deepcopy(x_0)

        f_k = torch.dot(plane_normal, x_0.contiguous().view(-1) - plane_point)
        sign_true = f_k.sign().item()

        beta = 0.001 * sign_true
        current_sign = sign_true

        while np.sign(current_sign) == np.sign(sign_true) and coord_vec.nonzero().size()[0] > 0 and sign_true != 0.0:

            f_k = torch.dot(plane_normal, x_i.contiguous().view(-1) - plane_point) + beta

            pert = f_k.abs() / coord_vec.abs().max()

            mask = torch.zeros_like(coord_vec)
            mask[np.unravel_index(torch.argmax(coord_vec.to('cpu').abs()), input_shape)] = 1.

            r_i = torch.clamp(pert, min=1e-4) * mask * coord_vec.sign()

            x_i = x_i - sign_true*r_i

            x_i = self.clip_image_values(x_i, min, max)

            f_k = torch.dot(plane_normal, x_i.contiguous().view(-1) - plane_point)
            current_sign = f_k.sign().item()

            coord_vec[r_i != 0] = 0

        return x_i

    def deepfool_fast_onecate(self, images, paras, label_loca_ori, avoid_cates, lambda_=2.):

        max_0 = torch.max(images.data[:, 0, :, :])
        max_1 = torch.max(images.data[:, 1, :, :])
        max_2 = torch.max(images.data[:, 2, :, :])
        min_0 = torch.min(images.data[:, 0, :, :])
        min_1 = torch.min(images.data[:, 1, :, :])
        min_2 = torch.min(images.data[:, 2, :, :])

        x_adv = torch.autograd.Variable(images.data, requires_grad=True)
        r_tot = torch.zeros_like(images).to(x_adv.device)

        hm_ori = self.model(images)[-1]['hm']
        hm_ori_sig = hm_ori.sigmoid()

        no_label = 0
        valid_cates = 0

        ################################################################################
        # Deep fool iters
        if len(label_loca_ori) != 0:
            valid_cates = set(label_loca_ori[:, 0].to('cpu').numpy())
            for iter in range(paras['iters_deepfool']):

                x_adv_temp = torch.autograd.Variable(x_adv.data, requires_grad=True)
                hm_adv = self.model(x_adv_temp)[-1]['hm']
                hm_adv_sig = hm_adv.sigmoid()

                label_loca_ori_temp = list()
                attack_count = 0
                for index, item in enumerate(label_loca_ori):
                    if hm_adv_sig[:, :, item[1], item[2]].argmax() != item[0]:
                        attack_count += 1
                    else:
                        label_loca_ori_temp.append(item.to('cpu').numpy())

                if attack_count == len(label_loca_ori):
                    break

                logits = hm_adv[:, :, label_loca_ori[:, 1], label_loca_ori[:, 2]].sum(dim=2)
                logits_ori = logits[:, list(valid_cates)].sum(dim=1)
                logits_adv = (logits - logits_ori)[0]

                weight_ori = torch.zeros_like(x_adv.data)

                for ori_label in valid_cates:
                    self.model.zero_grad()
                    if x_adv_temp.grad is not None:
                        x_adv_temp.grad.data.fill_(0)
                    hm_adv[:, ori_label, label_loca_ori[:, 1], label_loca_ori[:, 2]].sum().contiguous().backward(retain_graph=True)
                    weight_ori = weight_ori + x_adv_temp.grad.data

                pert = torch.Tensor([np.inf])[0].to(x_adv_temp.device)

                label_list = (np.array(logits.data.cpu().numpy().flatten())).flatten().argsort()[::-1]
                weight = torch.zeros_like(weight_ori)

                i_count = 0
                i = 0
                while i_count < paras['targets']:
                    search_flag = False
                    for search_item in avoid_cates:
                        if label_list[i] == search_item:
                            search_flag = True
                    if search_flag:
                        i += 1
                        continue

                    label = label_list[i]
                    self.model.zero_grad()
                    if x_adv_temp.grad is not None:
                        x_adv_temp.grad.data.fill_(0)
                    hm_adv[:, label, label_loca_ori[:, 1], label_loca_ori[:, 2]].sum().contiguous().backward(retain_graph=True)
                    weight_temp = x_adv_temp.grad.data - weight_ori
                    pert_k = torch.abs(logits_adv[label]) / weight_temp.norm()

                    if pert_k < pert:
                        pert = pert_k + 0.
                        weight = weight_temp + 0.
                        no_label = label
                    i_count += 1
                    i += 1

                r_i = torch.clamp(pert, min=1e-4) * weight / weight.norm()
                attack_rate = attack_count/len(label_loca_ori)

                x_adv.data += copy.deepcopy(r_i.data)
                x_adv.data[:, 0, :, :].clamp_(min_0, max_0)
                x_adv.data[:, 1, :, :].clamp_(min_1, max_1)
                x_adv.data[:, 2, :, :].clamp_(min_2, max_2)
                x_adv = torch.autograd.Variable(x_adv.data, requires_grad=True)

                r_tot = r_tot + copy.deepcopy(r_i.data)

                if attack_rate >= 0.99:
                    break

        # Normal Vector #############################################

        x_adv_temp = torch.autograd.Variable(x_adv.data, requires_grad=True)
        hm_adv = self.model(x_adv_temp)[-1]['hm']

        logits = hm_adv[:, :, label_loca_ori[:, 1], label_loca_ori[:, 2]].sum(dim=2)

        logits_ori = logits[:, list(valid_cates)].sum(dim=1)
        logits_adv = (logits - logits_ori)[0]

        logits_adv[no_label].backward(retain_graph=True)

        grad = copy.deepcopy(x_adv_temp.grad.data)
        grad = grad / grad.norm()

        x_adv = images + lambda_*r_tot

        return x_adv.data, r_tot.data, grad.data

    def gen_mask(self, hm_sig, valid_cates, paras):

        mask_set = {}
        noise_radius = 8

        for cate in valid_cates:

            temp_mask = torch.zeros_like(hm_sig)

            label_loca = (hm_sig[:, cate, :, :] > paras['vis_thresh']).nonzero()

            for index, item in enumerate(label_loca):
                temp_noise_radius = noise_radius
                for i in range(3):
                    try:
                        temp_mask[:, cate, item[1] - temp_noise_radius:item[1] + temp_noise_radius,
                        item[2] - temp_noise_radius:item[2] + temp_noise_radius] = 1.0
                        break
                    except:
                        temp_noise_radius //= 2

            mask_set[cate] = temp_mask.data

        return mask_set

    @staticmethod
    def clip_image_values(x, minv, maxv):
        x = torch.max(x, minv)
        x = torch.min(x, maxv)
        return x
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
        else:
            features, output = self.extractor(input)
            #  ModelOutputs()的返回值   return target_activations, x
            #  target_activations 代表 self.feature_module整层的参数  x代表ModelOutputs()中完整正向传播的结果，即output
            #  千分类 output size=[1,1000] 值为每类的概率值
        if index == None:
            # index 为目标类  如果没有人为设置，即为图片预测最大概率代表的标签 即分类器判断某张图片为某类的依据
            # ————————————和图片原本标签有区别——————————————————
            index = np.argmax(output.cpu().data.numpy())
        # output 为千分类器 的各类概率
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1  # one_hot size (1, 1000) 只有index下标元素为 1 其余都为0

        one_hot = torch.from_numpy(one_hot).requires_grad_(True)# 转tensor
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)

        else:
            # print(one_hot * output)
            one_hot = torch.sum(one_hot * output)
            # 得到 index 类的独热 概率  gradcam只传递一类的概率
        # 传某类的梯度

        self.feature_module.zero_grad()
        self.model.zero_grad()
        # 将model和feature_module中的梯度清零
        one_hot.backward(retain_graph=True)
        # 传递index 类的梯度

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        #print('grad size {}'.format(grads_val.shape))
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
        #print('weights size {}'.format(weights.shape))
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        #print(cam.shape) # (224, 224)
        for i, w in enumerate(weights):
            # 循环的是channel数 将所有channel中的（梯度*系数）相加
            # 而weights来自于target_layer的梯度
            cam += w * target[i, :, :]  # w grad      target  参数

        cam = np.maximum(cam, 0) # 去除负的元素
        cam = cv2.resize(cam, input.shape[2:]) #直接resize至原始图片尺寸
        cam = cam - np.min(cam)  # 归一化
        cam = cam / np.max(cam)
        return cam