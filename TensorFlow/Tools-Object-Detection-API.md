# Tensorflow Object Detection API  (Faster R-CNN with Resnet 101)

- ~~[Basic Tutorial](https://github.com/tensorflow/models/blob/master/object_detection/object_detection_tutorial.ipynb)~~: Jupyter, 학습된 모델을 Load하여 이미지 내 물체 예측 

    - ~~[Real-Time Object Recognition App with Tensorflow and OpenCV](https://github.com/datitran/Object-Detector-App)~~: 최적화 코드 


- [Quick Start: Distributed Training on the Oxford-IIIT Pets Dataset on Google Cloud](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/running_pets.md):   object detector 학습 방법 소개, **Transfer Learning**
    
    - ~~[How to train your own Object Detector with TensorFlow’s Object Detector API](https://medium.com/towards-data-science/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)~~

- [Configuring the Object Detection Training Pipeline](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/configuring_jobs.md): 학습 관련 설정 변경 



# 1. 개요
> 출처 : [홈페이지](https://github.com/tensorflow/models/tree/master/research/object_detection), [GOOD to GREAT](http://goodtogreate.tistory.com/entry/Tensorflow-Object-Detection-API-SSD-FasterRCNN)

h


2017.06.15월 공개 

지원 모델 
- Single Shot Multibox Detector (SSD) with MobileNet
- SSD with Inception V2
- Region-Based Fully Convolutional Networks (R-FCN) with ResNet 101
- Faster R-CNN with Resnet 101
- Faster RCNN with Inception Resnet v2

# 2. 설치 

> 설치 환경 ubunutu 16.4, python3, tf 1.2


###### 2.1 설치 

필수 패키지 (TF)
```bash
# For CPU
pip install tensorflow
# For GPU
#pip install tensorflow-gpu
conda install tensorflow-gpu
```

관련 패키지 설치
```bash
sudo apt-get install protobuf-compiler python-pil python-lxml
sudo pip install matplotlib pillow lxml
```

소스 다운로드
```bash
git clone https://github.com/tensorflow/models.git
```

###### 2.2 설정
- Protobuf 컴파일

```bash
# From tensorflow/models/
protoc object_detection/protos/*.proto --python_out=.
```

> Protobuf : XML과 같이 데이터를 저장하는 하나의 포맷, 컴파일후 언어에 맞는 형태의 데이타 클래스 파일을 생성 [[참고]](http://bcho.tistory.com/1182)

- Add Libraries to PYTHONPATH : slim 디렉터리를 append시키기 위함이다.

```bash
# From tensorflow/models/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

###### 2.3 설치 확인

```bash
python object_detection/builders/model_builder_test.py

.......
----------------------------------------------------------------------
Ran 7 tests in 0.013s

OK
```



## 3. 모델 생성 

[New Model 생성](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/defining_your_own_model.md)

## 4. Testing 

> 튜토리얼 : [./object detection/object_detection_tutorial.ipynb](https://github.com/tensorflow/models/blob/master/object_detection/object_detection_tutorial.ipynb)





###### [참고] 오류 

- TypeError: a bytes-like object is required, not 'str'
    - [TF 1.2이상으로 업그레이드](https://github.com/datitran/Object-Detector-App/issues/2)

export_inference_graph.py 파일 실행시 필요한 parameter 의 이름이 바뀌었습니다. 
- checkpoint_path -> trained_checkpoint_prefix 
- inference_graph_path -> output_directory 
- `python object_detection/export_inference_graph \` 가 아니라 `python object_detection/export_inference_graph.py \` 입니다.



## 5. Training 

> [How to train your own Object Detector with TensorFlow’s Object Detector API](https://medium.com/towards-data-science/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)


### 5.1 데이터 준비 

TFRecord을 입력으로 사용함 (eg.  PASCAL VOC datasetZ)

    - images.tar.gz : 이미지(JPG, PNG)
    
    - annotations.tar.gz : LIST(X_min, Y_min, X_max, Y_max) + (Label)
    
    ![](http://i.imgur.com/HfGjktp.png)

> 참고 : [tfrecord 파일 읽고 쓰기](http://bcho.tistory.com/1190)
    
###### Step 1. 이미지 준비 

- Google 이미지 검색 등 
      
###### Step 2. 수작업으로 라벨링 진행 

- 라벨링 툴(eg. [LanelImg](https://github.com/tzutalin/labelImg)) 이용

- PASCAL형태의 XML로 저장 


###### Step 3. Convert Tools 이용 TFRecord 변경 
                
- [`create_pascal_tf_record.py`](https://github.com/tensorflow/models/blob/master/object_detection/create_pascal_tf_record.py)

``` bash
# From tensorflow/models
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
python object_detection/create_pascal_tf_record.py \
    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2012 --set=train \
    --output_path=pascal_train.record
python object_detection/create_pascal_tf_record.py \
    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2012 --set=val \
    --output_path=pascal_val.record
```

결과물 
- *_train.record 
- *_val.record
- [label_map.pbtxt](https://github.com/tensorflow/models/tree/master/object_detection/data)


    
###### [참고] 자신만의 Convert Tool 만들기 

- [using_your_own_dataset.md](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/using_your_own_dataset.md)
                        
- [적용 예 Raccoon 이미지를 변환 하는스크립트(XML - CSV - TFRecord)](https://github.com/datitran/raccoon-dataset)  

 
###### Step 4. 작업 위치로 이동 
    
        
- 저장 위치 : `tensorflow/models`

```bash
- images.tar.gz
- annotations.tar.gz
+ images/
+ annotations/
+ object_detection/
... other files and directories
```
    
> 이미지 크기는 300~500 pixels추천(???) -> OOM문제 발생, Batch-size조절로 가능 



### 5.2 Config 파일 수정 

*.Config파일에 `model parameters`, `training parameters` and `eval parameters` 모두 포함하고 있음 

주요 설정 항목 
- num_class : eg. 클래스가 하나 이면 1
- PATH : Train data PATH, Test data PATH, label map PATH
    - label map : *.pbtxt파일, id + name 으로 구성 (중요 : id는 항상 1부터 시작)


> [Configuring the Object Detection Training Pipeline](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/configuring_jobs.md), [[Sample config 파일]](https://github.com/tensorflow/models/tree/master/object_detection/samples/configs)



### 5.3 실행 

#### A. Local 학습 

> [Running Locally](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/running_locally.md)

```bash
# Recommended Directory Structure for Training and Evaluation
+data
  -label_map file
  -train TFRecord file
  -eval TFRecord file
+models
  + model
    -pipeline config file
    +train
    +eval
```

###### Running the Training Job
```bash
# From the tensorflow/models/ directory
python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \  
    --train_dir=${PATH_TO_TRAIN_DIR}
```
- `${PATH_TO_YOUR_PIPELINE_CONFIG}` : the pipeline config
- `${PATH_TO_TRAIN_DIR}` : the directory in which training checkpoints and events will be written to.

> 사용자가 중단하기 전까지 계속 학습 수행 


###### Running the Evaluation Job
```bash
# From the tensorflow/models/ directory
python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --checkpoint_dir=${PATH_TO_TRAIN_DIR} \
    --eval_dir=${PATH_TO_EVAL_DIR}
```
- `${PATH_TO_YOUR_PIPELINE_CONFIG}` : the pipeline config
- `${PATH_TO_TRAIN_DIR}` : the directory in which training checkpoints were saved
- `${PATH_TO_EVAL_DIR}` :the directory in which evaluation events will be saved


###### Running Tensorboard
```bash
tensorboard --logdir=${PATH_TO_MODEL_DIRECTORY}
```

- `${PATH_TO_MODEL_DIRECTORY}` : The directory that contains the train and eval directories





#### B. Cloud 학습 

- [Running on Google Cloud Platform](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/running_on_cloud.md)

- [Starting Training and Evaluation Jobs on Google Cloud ML Engine](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/running_pets.md#starting-training-and-evaluation-jobs-on-google-cloud-ml-engine)

### 5.4 Export Model 

- 학습시 생성된 `checkpoint`파일을 `Tensorflow graph proto` 형태로 export가능 

- 입력 : checkpoint files 
    - model.ckpt-${CHECKPOINT_NUMBER}.data-00000-of-00001
    - model.ckpt-${CHECKPOINT_NUMBER}.index
    - model.ckpt-${CHECKPOINT_NUMBER}.meta

- 출력 : output_inference_graph.pb


```python
# From tensorflow/models
python object_detection/export_inference_graph \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --checkpoint_path model.ckpt-${CHECKPOINT_NUMBER} \
    --inference_graph_path output_inference_graph.pb
```



## 6. Transfer Learning

### 6.1 새 학습 데이터 + 학습된(Pre Trained) 모델 준비 

#### A. 새 학습 데이터

- [4.1] 참고 


#### B. 학습된 모델 

model.ckpt* 파일 다운 받기 

```
-rw-r----- 1 hjlim99 hjlim99 188M Jun 12 00:58 frozen_inference_graph.pb
-rw-r----- 1 hjlim99 hjlim99  20M Jun 12 00:58 graph.pbtxt
-rw-r----- 1 hjlim99 hjlim99 426M Jun 12 01:00 model.ckpt.data-00000-of-00001
-rw-r----- 1 hjlim99 hjlim99  40K Jun 12 01:00 model.ckpt.index
-rw-r----- 1 hjlim99 hjlim99  11M Jun 12 01:00 model.ckpt.meta
```

- `graph.pbtxt` : a graph proto
- `model.ckpt.data-00000-of-00001`, `model.ckpt.index`, `model.ckpt.meta` : **a checkpoint** 
- `frozen_inference_graph.pb` : a frozen graph proto with weights baked into the graph as constants 

> [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md)

### 6.2 Training Config 파일 수정 

- [4.2] 참고 


```~~~~
# 실행 단계에 필요한 파일들 
  + data/
    - faster_rcnn_resnet101_pets.config
    - model.ckpt.index
    - model.ckpt.meta
    - model.ckpt.data-00000-of-00001
    - pet_label_map.pbtxt
    - pet_train.record
    - pet_val.record
```

### 6.3 학습 실행

- [4.3] 참고 

### 6.4 Export Model

---
수정 
/models/object_detection/object_detection_tutorial.ipynb#


```
#MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'
```

```
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'frame{}.jpg'.format(i)) for i in range(1001, 11565) ]

```

```
import scipy.misc

#plt.figure(figsize=IMAGE_SIZE)
#plt.imshow(image_np)
print(image_path)  
#cv2.imshow(image_np)
#plt.savefig('./save/{}.png'.format(image_path))


image = scipy.misc.toimage(image_np)
image.save('./save/{}.jpg'.format(image_path))

```
- 결과물 저장 위치 :`/workspace/models/object_detection/save/test_images`