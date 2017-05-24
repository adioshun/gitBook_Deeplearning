|년도|알고리즘|링크|입력|출력|특징|
|-|-|-|-|-|-|
|2014|R-CNN|[논문](https://arxiv.org/abs/1311.2524)|Image|Bounding boxes + labels for each object in the image.|AlexNet, 'Selective Search'사용 |
|2015|Fast R-CNN|[논문](https://arxiv.org/abs/1504.08083)|Images with region proposals.|Object classifications |Speeding up and Simplifying R-CNN, RoI Pooling|
|2016|Faster R-CNN|[논문](https://arxiv.org/abs/1506.01497),[한글](https://curt-park.github.io/2017-03-17/faster-rcnn/)| CNN Feature Map.|A bounding box per anchor|MS, Region Proposal|
|2017|Mask R-CNN|[논문](https://arxiv.org/abs/1703.06870)|CNN Feature Map.|Matrix with 1s on all locations|Facebook, pixel level|

> 출처 : [A Brief History of CNNs in Image Segmentation: From R-CNN to Mask R-CNN](https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4)



|R-CNN Series|YOLO/SSD|
|-|-|
|2-step|1-Step Process|
|process of region proposal and object recognition|Do region proposal and classification at same time|

> 출처 : [Recent Progress on Object Detection_20170331](https://www.slideshare.net/JihongKang/recent-progress-on-object-detection20170331)

# YOLO

- Super fast detector (21~155 fps)

- Finding objects at each grid __in parallel__

- 성능 : Fast R-CNN < YOLO < Faster R-CNN 




# SSD 

- Faster R-CNN + YOLO

- Multi-scale feature map detection 
    - Detect small objects on lower level, large objects on higher level
    
- End-to-End training/testing 


