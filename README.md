# Boundary-Aware Set Abstraction for 3D Object Detection
By Huang Zhe, Wang YongCai, Tang XinGui, and Sun HongYu

![backbone](https://user-images.githubusercontent.com/44192081/195514096-248fe526-d9b3-486f-8ba9-4c2145086457.png)
## abstract
The basic components of a point-based 3D object detector is set abstraction (SA) layer, which  downsamples points for better efficiency and enlarges receptive fields. However, existing SA layer only takes the relative locations among points into consideration, e.g. using furthest point sampling,  while ignoring point features. Because the points on the objects take small proportion of space, uniform and cascaded SA may 
don't contain objects' points in the last layer, degrading 3D object detection performances. We are thus motivated to design a new lightweight and effective SA layer named Boundary-Aware Set Abstraction layer  (BA-Net) to retain important foreground and boundary points during cascaded down-sampling. Technically, we first embed a lightweight point segmentation model (PSM) to compute the point-wise foreground scores, then propose a Boundary Prediction Model(BPM) to detect points on object boundaries.  Finally,  point scores are used to twist inter-node distances and furthest point down-sampling is conducted in the twisted distance space (B-FPS). We experiment  on  KITTI dataset and the results show that BA-Net improves detection performance especially in harder cases. Additionally, BA-Net is easy-to-plug-in point-based module and able to boost various detectors. 

![BA-Net](https://user-images.githubusercontent.com/44192081/195514333-1c6ca613-44dd-4938-9fec-00447fa1ce0b.png)

## Main Result

<img width="1012" alt="image" src="https://user-images.githubusercontent.com/44192081/195514650-dd5f7ee2-400b-4dc8-8112-47596174237d.png">
