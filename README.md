# Boundary-Aware Set Abstraction for 3D Object Detection
By Huang Zhe, Wang YongCai, Tang XinGui, and Sun HongYu


<img src="https://user-images.githubusercontent.com/44192081/195514096-248fe526-d9b3-486f-8ba9-4c2145086457.png" width="70%">

## abstract
The basic components of a point-based 3D object detector is set abstraction (SA) layer, which  downsamples points for better efficiency and enlarges receptive fields. However, existing SA layer only takes the relative locations among points into consideration, e.g. using furthest point sampling,  while ignoring point features. Because the points on the objects take small proportion of space, uniform and cascaded SA may 
don't contain objects' points in the last layer, degrading 3D object detection performances. We are thus motivated to design a new lightweight and effective SA layer named Boundary-Aware Set Abstraction layer  (BA-Net) to retain important foreground and boundary points during cascaded down-sampling. Technically, we first embed a lightweight point segmentation model (PSM) to compute the point-wise foreground scores, then propose a Boundary Prediction Model(BPM) to detect points on object boundaries.  Finally,  point scores are used to twist inter-node distances and furthest point down-sampling is conducted in the twisted distance space (B-FPS). We experiment  on  KITTI dataset and the results show that BA-Net improves detection performance especially in harder cases. Additionally, BA-Net is easy-to-plug-in point-based module and able to boost various detectors. 


<img src="https://user-images.githubusercontent.com/44192081/195514333-1c6ca613-44dd-4938-9fec-00447fa1ce0b.png" width="80%">


## Main Result

Method | Easy | Mod. | Hard | mAP
--------- | --------- | ------------- | ------------- | ------------- |
PointRCNN|91.57| 82.24 |80.45| 84.75
PointRCNN+BA-Net|**+0.75**| **+0.8**| **+1.86** |**+1.14**
3DSSD|91.54 |83.46 |82.18 |85.73
3DSSD+BA-Net|**+0.89** |**+1.93** |**+0.38**| **+1.06**


## Usage: Preparation
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 14.04/16.04/18.04/20.04/21.04)
* Python 3.6+
* PyTorch 1.1 or higher (tested on PyTorch 1.1, 1,3, 1,5~1.10)
* CUDA 9.0 or higher (PyTorch 1.3+ needs CUDA 9.2+)
* spconv v1.0 (commit 8da6f96) or spconv v1.2 or spconv v2.x

## Building Kernel
NOTE: Please re-install pcdet v0.5 by running python setup.py develop
```
git clone https://github.com/HuangZhe885/Boundary-Aware-SA.git
cd Boundary-Aware-SA
pip install -r requirements.txt 
python setup.py develop 
```
install spconv

```
git clone https://github.com/traveller59/spconv.git --recursive
cd spconv
python setup.py bdist_wheel
cd ./dist
pip install *
```

## Dataset

Please download the official [KITTI 3D object detection dataset](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize the downloaded files as follows (the road planes could be downloaded from [road plane], which are optional for data augmentation in the training):
* Generate the data infos by running the following command:
```
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

# Training & Testing
## Train a model
You could optionally add extra command line parameters --batch_size ${BATCH_SIZE} and --epochs ${EPOCHS} to specify your preferred parameters.

```
python train.py --cfg_file ${CONFIG_FILE}
```
## Test a model
```
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --eval_all
```
## Visualization


Visualizing detection results on KITTI val split. The ground truth and predictions are labeled in red and green respectively. Pink points mark the 512 key points sampled in last SA layer.

Harder instances contain fewer LiDAR points and are not likely to be selected, therefore, it is difficult for them to survive in the vanilla FPS down-sampling, and the features for remote (or small) instances cannot be fully transmitted to the next layer of the network, while BA-Net
can keep adequate interior boundary points of different foreground instances. It preserves rich information for regression and classification
Here we present experimental results evaluated on the KITTI validation set.
<img width="785" alt="image" src="https://user-images.githubusercontent.com/44192081/195516445-83972293-71b2-476b-8217-7532d3cafebd.png">



Snapshots of our 3D detection results on row 1 (left is 3DSSD, right is BA-Net) on the KITTI validation set. The predicted bounding boxes are shown in green, and are project back onto the color images in pink (2th rows) for visualization.

<img src="https://user-images.githubusercontent.com/44192081/195514942-9f0f384e-7fac-4677-8212-9d85ad3eb2b1.png" width="50%">

## Acknowledgement

This project is built with [OpenPCDet](https://github.com/blakechen97/SASA/blob/main/OpenPCDet.md), a powerful toolbox for LiDAR-based 3D object detection. Please refer to OpenPCDet.md and the official [github repository](https://github.com/open-mmlab/OpenPCDet) for more information.


