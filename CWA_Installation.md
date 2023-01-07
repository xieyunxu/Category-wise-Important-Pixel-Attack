#Introduction
This repository contains the codes for Sparse Category Attack and Dense Category Attack. The paper of Category-wise Attack has been submit to CVPR 2020.

#5843: Category-wise Attack: Transferable Adversarial Examples for Anchor Free Object Detection

Implementation

CenterNet

At first, you need to install the CenterNet and run successfully on MS-COCO and PascalVOC.

Objects as Points,
Xingyi Zhou, Dequan Wang, Philipp Krähenbühl,
https://github.com/xingyizhou/CenterNet
Install SCA and DCA

Then, copy three *.py file into CenterNet-master/src/lib/detectors/. and open CenterNet-master/src/lib/detectors/detector_factory.py, and change the

from .ctdet import CtdetDetector
into DCA

from .ctdet_DCA import CtdetDetector
or SCA

from .ctdet_SCA import CtdetDetector
Finally, run the CenterNet as before.

Reproducibility

In all experiment, the parameter of DCA is the same. You can directly use the ctdet_DCA.py to reproducing all results of DCA.

You can also reproducing qualitative results of SCA by directly using the ctdet_SCA.py. In the code of SCA we provide:

paras['epsilon'] = 0.05
If you want to reproducing other results of SCA, such as white-box attack, black-box attack and perceptibility, please set as following:

paras['epsilon'] = 1.0
Then, you can reproducing other results of SCA.
