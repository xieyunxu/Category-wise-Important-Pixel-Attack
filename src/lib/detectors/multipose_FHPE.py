from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

try:
    from external.nms import soft_nms_39
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import multi_pose_decode
from models.utils import flip_tensor, flip_lr_off, flip_lr
from utils.image import get_affine_transform
from utils.post_process import multi_pose_post_process
from utils.debugger import Debugger

from .base_detector_attack_release import BaseDetector


class MultiPoseDetector(BaseDetector):
    def __init__(self, opt):
        super(MultiPoseDetector, self).__init__(opt)
        self.flip_idx = opt.flip_idx

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
        eps = 0.05 * max_pixel# 0.05
        iters = 3# 10
        cates = 17
        attack_thres = 0.1 # 0.1
        eps_each_iter = eps/iters
        x_adv = torch.autograd.Variable(images.data, requires_grad=True)

        crit = torch.nn.CrossEntropyLoss()

        out = self.model(images)[-1]
        #hm_ori = out['hm'] # ['hm'] ['hm_hp']
        str_name = 'hm_hp'
        hmhp_ori = out[str_name]
        #heads {'hm': 1, 'wh': 2, 'hps': 34, 'reg': 2, 'hm_hp': 17, 'hp_offset': 2}
        #第一阶段：
        #hm: 用于寻找第一阶段keypoint 的 heatmap 只有(H/4)*(W/4)*1   只找一个点，因为只有human类
        #hps: 第一阶段centerpoint回归得到17个点的偏移量(17*2)

        #第二阶段：
        #hm_hp: 第二部对17个点的预测
        #wh: 第一阶段kp(centerpoint)所附带的长和宽 姿态估计带有目标检测,只有落在bb里的点可以为joint

        hm_ori_sig = hmhp_ori.sigmoid()


        noise_tot = torch.zeros_like(x_adv.data)

        for iter in range(iters):
            x_adv_temp = torch.autograd.Variable(x_adv.data, requires_grad=True)

            hm_adv = self.model(x_adv_temp)[-1][str_name]
            hm_adv_sig = hm_adv.sigmoid()

            label_temp = torch.LongTensor([0]).to(torch.device('cuda'))
            loss = crit(hm_adv[:, :, 0, 0], label_temp) - crit(hm_adv[:, :, 0, 0], label_temp)

            noise = torch.zeros_like(x_adv_temp)

            #for cate in range(cates-1,cates): # range 左闭又开 if cates=1 cate=0 only
            for cate in range(cates):  # range 左闭又开 if cates=1 cate=0 only

                label_loca_adv = (hm_adv_sig[:, cate, :, :] > attack_thres).nonzero()
                # 用于寻找每层中的位置,生成一个横坐标和纵坐标集合

                label_temp = torch.LongTensor([cate]).to(torch.device('cuda'))

                if len(label_loca_adv) == 0:
                    continue

                loss_count = 0
                for index, item in enumerate(label_loca_adv):
                    # 对每层中的每个位置叠加梯度
                    if hm_ori_sig[:, cate, item[1], item[2]] > attack_thres:
                        # item[1]和item[2]为非零元素横纵坐标 item[0]

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

            noise += x_adv_temp.grad / x_adv_temp.grad.abs().max()

            noise_tot = noise_tot + (noise.data.mul(eps_each_iter)).data

            x_adv.data += noise.data.sign().mul(eps_each_iter)
            x_adv.data[:, 0, :, :].clamp_(min_0, max_0)
            x_adv.data[:, 1, :, :].clamp_(min_1, max_1)
            x_adv.data[:, 2, :, :].clamp_(min_2, max_2)
            x_adv = torch.autograd.Variable(x_adv.data, requires_grad=True)

        ################################################################################
        print("FHPE/PHPE pose")
        print('noise_tot shape: ', noise_tot.shape)
        noise_total_ = noise_tot[0].cpu().numpy()
        p2 = (np.linalg.norm(noise_total_[0]) + np.linalg.norm(noise_total_[1]) + np.linalg.norm(noise_total_[2])) / (
                    512 * 512)
        print('L2 norm of pertubation:{} '.format(p2))
        count = np.count_nonzero(noise_total_)
        print('L0 norm of pertubation:{} '.format(count / (3 * 512 * 512)))

        output = self.model(x_adv)[-1]
        #hm = output['hm']
        #wh = output['wh']
        #hps = output['hps']
        #reg = output['reg']
        #hm_hp = output['hm_hp']
        #hp_offset = output['hp_offset']
        #print('hm shape:{}'.format(hm.shape))
        #print('wh shape:{}'.format(wh.shape))
        #print('hps shape:{}'.format(hps.shape))
        #print('reg shape:{}'.format(hps.shape))
        #print('hm_hp shape:{}'.format(hps.shape))
        #print('hp_offset shape:{}'.format(hps.shape))
        output['hm'] = output['hm'].sigmoid_()
        if self.opt.hm_hp and not self.opt.mse_loss:
            output['hm_hp'] = output['hm_hp'].sigmoid_()

        reg = output['reg'] if self.opt.reg_offset else None
        hm_hp = output['hm_hp'] if self.opt.hm_hp else None
        hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
        torch.cuda.synchronize()
        forward_time = time.time()

        if self.opt.flip_test:
            output['hm'] = (output['hm'][0:1] + flip_tensor(output['hm'][1:2])) / 2
            output['wh'] = (output['wh'][0:1] + flip_tensor(output['wh'][1:2])) / 2
            output['hps'] = (output['hps'][0:1] +
                             flip_lr_off(output['hps'][1:2], self.flip_idx)) / 2
            hm_hp = (hm_hp[0:1] + flip_lr(hm_hp[1:2], self.flip_idx)) / 2 \
                if hm_hp is not None else None
            reg = reg[0:1] if reg is not None else None
            hp_offset = hp_offset[0:1] if hp_offset is not None else None

        dets = multi_pose_decode(
            output['hm'], output['wh'], output['hps'],
            reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)

        if return_time:
            return output, dets, forward_time, x_adv, noise_tot
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets = multi_pose_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'])
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 39)
            # import pdb; pdb.set_trace()
            dets[0][j][:, :4] /= scale
            dets[0][j][:, 5:] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        results[1] = np.concatenate(
            [detection[1] for detection in detections], axis=0).astype(np.float32)
        if self.opt.nms or len(self.opt.test_scales) > 1:
            soft_nms_39(results[1], Nt=0.5, method=2)
        results[1] = results[1].tolist()
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        dets = dets.detach().cpu().numpy().copy()
        dets[:, :, :4] *= self.opt.down_ratio
        dets[:, :, 5:39] *= self.opt.down_ratio
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(((
                               img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
        pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hm')
        if self.opt.hm_hp:
            pred = debugger.gen_colormap_hp(
                output['hm_hp'][0].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hmhp')
    def debug_noise(self, debugger, images, dets, output, img_name, noise, scale=1):
        detection = dets.detach().cpu().numpy().copy()
        detection[:, :, :4] *= self.opt.down_ratio
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img  * self.std + self.mean) * 255).astype(np.uint8)

            p_mean = np.array([0.485, 0.456, 0.406],
                              dtype=np.float32).reshape(1, 1, 3)
            p_std = np.array([0.229, 0.224, 0.225],
                             dtype=np.float32).reshape(1, 1, 3)

            noise = noise[i].detach().cpu().numpy().transpose(1, 2, 0)
            #mask = cv2.imread('./mask/mask_13.jpg', 1)
            print(noise)
            noise = ((noise * 30 * p_std + p_mean)*255).astype(np.uint8)
            #mask = cv2.resize(mask, size)
            #mask = cv2.resize(mask,(512,512),interpolation=cv2.INTER_NEAREST)

            #mask = np.float32(mask)
            background = np.ones((512,512,3))*1
            background[:, :, 0] *= 119
            background[:, :, 1] *= 116
            background[:, :, 2] *= 103
            #cv2.imshow('1',mask)
            #cv2.waitKey(0)
            #print('mask',np.shape(mask))
            #noise = (noise * mask/255 + background*(255-mask)/255).astype(np.uint8)
            #print(noise)


            debugger.add_img(noise, img_id='noise_img_{:.1f}'.format(scale))

            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))

    def show_results(self, debugger,image, results):
        debugger.add_img(image, img_id='multi_pose')#在这里加入的
        for bbox in results[1]:
            if bbox[4] > self.opt.vis_thresh:
                # debugger.add_coco_bbox(bbox[:4], 0, bbox[4], img_id='multi_pose')
                debugger.add_coco_hp(bbox[5:39], img_id='multi_pose')
        debugger.show_all_imgs(pause=self.pause)
        debugger.save_all_imgs(path='D:/python/object_detection/CenterNet/outputs/attack/', genID=True)
    def save_noise(self, debugger,noise):
        debugger.add_img(noise, img_id='multi_pose')
        debugger.save_img(imgId='multi_pose_{:.1f}'.format(1),path='D:/python/object_detection/CenterNet/outputs/noise/')

    def save_hm(self, debugger, hm):
        total = hm[0]
        imgId = 'multi_pose'
        path = 'D:/python/object_detection/CenterNet/outputs/hm/'
        for i in range(hm.shape[0]):
            total = total + hm[i]
            cv2.imwrite(path + '{}_{}.png'.format(i,imgId),hm[i])
        cv2.imwrite(path + 'total.png', total)
