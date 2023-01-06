# Category-wise-Important-PIxel-Attack
# Getting Started
This document provides tutorials of Catergory-wise Attack (CWA) framework.
![](Category-wise%20Important%20Pixel%20Attack/fig1.png)
![](Category-wise%20Important%20Pixel%20Attack/fig2.png)
Following the procedures below:
1) The experiments of this work is based on the target anchor-free object detector CenterNet, more details can be found at [Object as Points](http://arxiv.org/abs/1904.07850)
2) The basic installation procedures of CenterNet, all about is here [CenterNet Installation Procedures](Category-wise%20Important%20Pixel%20Attack/INSTALL.md) 
3) Our work conducts experiments on PascalVOC and MS-COCO datasets, all relevant is here [Datasets Download](Category-wise%20Important%20Pixel%20Attack/DATA.md).
4)  Download the backbone  networks you want to evaluate from the [model zoo](Category-wise%20Important%20Pixel%20Attack/MODEL_ZOO.md) and move them in `Root_File/models/`. 
5) The installation procedures of CWA framework.


# Run experiments
### Detection
To evaluate PascalVOC object detection
~~~
> Clean outputs:
> run:
python test.py ctdet --exp_id pascal_dla_1x_clean --dataset pascal --arch dla_34 --not_prefetch_test --load_model ../models/ctdet_pascal_dla_512.pth
~~~
~~~
> Adversarial outputs (DCA-G):
> run:
python test_universal.py ctdet --exp_id pascal_dla_1x_Uniattack --dataset pascal --arch dla_34 --not_prefetch_test --load_model ../models/ctdet_pascal_dla_512.pth
> Adversarial outputs (DCA-L):
> run:
python test_universal.py ctdet --exp_id pascal_dla_1x_Uniattack --dataset pascal --arch dla_34 --not_prefetch_test --load_model ../models/ctdet_pascal_dla_512.pth
> Adversarial outputs (DCA-S):
> run:
python test_universal.py ctdet --exp_id pascal_dla_1x_Uniattack --dataset pascal --arch dla_34 --not_prefetch_test --load_model ../models/ctdet_pascal_dla_512.pth
~~~
### Pose estimation
~~~
> Clean outputs:
> run:
python test.py multi_pose --exp_id coco_resdcn18_clean --dataset coco_hp --arch dla_34 --not_prefetch_test --load_model ../models/multi_pose_dla_1x.pth
~~~


~~~
> Adversarial outputs (DCA-G):
> run:
python test_universal.py multi_pose --exp_id coco_resdcn18_attack_uni --dataset coco_hp --arch dla_34 --not_prefetch_test --load_model ../models/multi_pose_dla_1x.pth
> Adversarial outputs (DCA-L):
> run:
python test_universal.py ctdet --exp_id pascal_dla_1x_Uniattack --dataset pascal --arch dla_34 --not_prefetch_test --load_model ../models/ctdet_pascal_dla_512.pth
> Adversarial outputs (DCA-S):
> run:
python test_universal.py ctdet --exp_id pascal_dla_1x_Uniattack --dataset pascal --arch dla_34 --not_prefetch_test --load_model ../models/ctdet_pascal_dla_512.pth
~~~
