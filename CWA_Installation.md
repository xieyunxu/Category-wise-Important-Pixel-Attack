# Implementation
At first, you need to install the CenterNet and run successfully on MS-COCO and PascalVOC.

Objects as Points,
Xingyi Zhou, Dequan Wang, Philipp Krähenbühl,
https://github.com/xingyizhou/CenterNet
# Install CWA Family Attack
Open CenterNet-master/src/lib/detectors/detector_factory.py, and change the 
from .ctdet import CtdetDetector
into from .ctdet_DCA import CtdetDetector

Run the CenterNet as before.

Reproducibility

In all experiment, the parameter of DCA is the same. You can directly use the ctdet_DCA.py to reproducing all results of DCA.

You can also reproducing qualitative results of SCA by directly using the ctdet_SCA.py. In the code of SCA we provide:

paras['epsilon'] = 0.05
If you want to reproducing other results of SCA, such as white-box attack, black-box attack and perceptibility, please set as following:

paras['epsilon'] = 1.0
