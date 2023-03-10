# Installation
The code was tested on Ubuntu 16.04, with [Anaconda](https://www.anaconda.com/download) Python 3.6 and [PyTorch]((http://pytorch.org/)) v0.4.1. NVIDIA GPUs are needed for both training and testing.
After install Anaconda:
0. [Optional but recommended] create a new conda environment.
    ~~~
    conda create --name CenterNet python=3.6
    ~~~
    And activate the environment.
    
    ~~~
    conda activate CenterNet
    ~~~

1. Install pytorch0.4.1:
    ~~~
    conda install pytorch=0.4.1 torchvision -c pytorch
    ~~~
    
    And disable cudnn batch normalization(Due to [this issue](https://github.com/xingyizhou/pytorch-pose-hg-3d/issues/16)).
     ~~~
    # PYTORCH=/path/to/pytorch # usually ~/anaconda3/envs/CenterNet/lib/python3.6/site-packages/
    # for pytorch v0.4.0
    sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
    # for pytorch v0.4.1
    sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
     ~~~
     
     For other pytorch version, you can manually open `torch/nn/functional.py` and find the line with `torch.batch_norm` and replace the `torch.backends.cudnn.enabled` with `False`. We observed slight worse training results without doing so. 
     
2. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
    ~~~
    # COCOAPI=/path/to/clone/cocoapi
    git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
    cd $COCOAPI/PythonAPI
    make
    python setup.py install --user
    ~~~

3. Clone this repo:
    ~~~
    Root_File=/path/to/clone/CenterNet
    git clone https://github.com/xingyizhou/CenterNet Root_File
    ~~~


4. Install the requirements
    ~~~
    pip install -r requirements.txt
    ~~~
    
    
5. Compile deformable convolutional (from [DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_0.4)).
    ~~~
    cd $Root_File/src/lib/models/networks/DCNv2
    ./make.sh
    ~~~
6. [Optional, only required if you are using extremenet or multi-scale testing] Compile NMS if your want to use multi-scale testing or test ExtremeNet.
    ~~~
    cd $Root_File/src/lib/external
    make
    ~~~

7. Download pertained models for [object detection]() or [human pose estimation]() and move them to `$Root_File/models/`. 
   More models can be found in [Model zoo](Category-wise%20Important%20Pixel%20Attack/MODEL_ZOO.md).
   

8.  GCA Installation:
    Open  the file Root_File/src/lib/detectors/detector_factory.py
    ~~~
    > For object detection task:
    > replace 
    from .ctdet import CtdetDetector
    > using
    from .ctdet import CtdetDetector
    > that's the way obtaining the adversarial results instead of original clean one.
    ~~~
    
    ~~~
    > For pose estimation task:
    > replace 
    from .multi_pose import MultiPoseDetector
    > using
    from .multipose_uni import MultiPoseDetector
    ~~~