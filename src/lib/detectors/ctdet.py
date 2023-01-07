from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

try:
  from external.nms import soft_nms
except:
  print('NMS not imported! If you need it,'
        ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import ctdet_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector

class CtdetDetector(BaseDetector):
  def __init__(self, opt):
    super(CtdetDetector, self).__init__(opt)
  
  def process(self, images, return_time=False):
    with torch.no_grad():
      output = self.model(images)[-1]
      hm = output['hm'].sigmoid_()
      wh = output['wh']
      reg = output['reg'] if self.opt.reg_offset else None
      if self.opt.flip_test:
        hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
        wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
        reg = reg[0:1] if reg is not None else None
      torch.cuda.synchronize()
      forward_time = time.time()
      #hm为output中的heatmap
      #wh为每个点的长和宽
      dets,sc = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
      
    if return_time:
      return output, dets, forward_time
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

  def debug(self, debugger, images, dets, output, scale=1):
    detection = dets.detach().cpu().numpy().copy()
    detection[:, :, :4] *= self.opt.down_ratio
    hm = output['hm'][0].detach().cpu().numpy().transpose(1, 2, 0)#output['hm'][i].shape 80,128,128
    self.save_hm(hm)
    for i in range(1):
      #i=0
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      print(output['hm'][i].shape)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      #output['hm'][i].shape 80,128,128
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      for k in range(len(dets[i])):
        if detection[i, k, 4] > self.opt.center_thresh:
          debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                 detection[i, k, 4], 
                                 img_id='out_pred_{:.1f}'.format(scale))
  '''
  #useful
  
    for i in range(80):
      sc = hm[i].reshape(1, 128,128)
      img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      pred = debugger.gen_colormap(sc.detach().cpu().numpy())
      #debugger.add_blend_img(img, pred, 'pred_hm+ori_layer{}'.format(i))
      #这个可以将heatmap与原图合成
      debugger.add_img(pred, 'pred_heatmap_layer{}'.format(i))
      # 这个直接将每类的heatmap直接输出
'''

  def show_results(self, debugger, image, results):
    debugger.add_img(image, img_id='ctdet')
    for j in range(1, self.num_classes + 1):
      for bbox in results[j]:
        if bbox[4] > self.opt.vis_thresh:#只有大于vis_thresh才会加入add_coco_bbox
          debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
    debugger.show_all_imgs(pause=self.pause)
    #debugger.show_all_imgs('$CenterNet_ROOT/outputs/',genID=True)
    debugger.save_all_imgs(path='D:/python/object_detection/CenterNet/outputs/ctdet_clean/',genID=True)

  def save_hm(self, hm):
    total = hm[:,:,0]
    print('hm_size: ',hm.shape)
    imgId = 'ctdet'
    path = 'D:/python/object_detection/CenterNet/outputs/hm/'
    for i in range(hm.shape[2]):
     # print('heat layer[{}] property max:{}, min:{}, mean:{}'.format(i,np.max(hm[:,:,i]),np.min(hm[:,:,i]),np.sum(hm[:,:,i])))
      heatmap = cv2.applyColorMap((hm[:,:,i]*2*255).astype(np.uint8),cv2.COLORMAP_HOT)
      total = total + hm[:,:,i]
      cv2.imwrite(path + '{}_{}.png'.format(i, imgId),heatmap)
    cv2.imwrite(path + 'total.png',total)


