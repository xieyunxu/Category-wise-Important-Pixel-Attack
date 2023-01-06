# Dataset preparation
### COCO
- Download the images (2017 Train, 2017 Val, 2017 Test) from [coco website](http://cocodataset.org/#download).
- Download annotation files (2017 train/val and test image info) from [coco website](http://cocodataset.org/#download). 
- Place the data (or create symlinks) to make the data folder like:
  ~~~
  ${ROOT_File}
  |-- data
  `-- |-- coco
      `-- |-- annotations
          |   |-- instances_train2017.json
          |   |-- instances_val2017.json
          |   |-- person_keypoints_train2017.json
          |   |-- person_keypoints_val2017.json
          |   |-- image_info_test-dev2017.json
          |---|-- train2017
          |---|-- val2017
          `---|-- test2017
  ~~~
### Pascal VOC
- Run
    ~~~
    cd $ROOT_File/tools/
    bash get_pascal_voc.sh
    ~~~
- The above script includes:
    - Download, unzip, and move Pascal VOC images from the [VOC website](http://host.robots.ox.ac.uk/pascal/VOC/). 
    - [Download](https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip) Pascal VOC annotation in COCO format (from [Detectron](https://github.com/facebookresearch/Detectron/tree/master/detectron/datasets/data)). 
    - Combine train/val 2007/2012 annotation files into a single json. 
  
- Move the created `voc` folder to `data` (or create symlinks) to make the data folder like:
  ~~~
  ${ROOT_File}
  |-- data
  `-- |-- voc
      `-- |-- annotations
          |   |-- pascal_trainval0712.json
          |   |-- pascal_test2017.json
          |-- images
          |   |-- 000001.jpg
          |   ......
          `-- VOCdevkit
  
  ~~~


