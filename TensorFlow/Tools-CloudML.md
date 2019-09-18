# CLOUD MACHINE LEARNING ENGINE


## 1. 환경 설정 

### 1. 1 Cloud Shell

- 구글 콘솔창 우측 상단의 `웹 기반 shell`활용 

- 미리 모든 패키지가 설치되어 있음 

### 2.2 Local(MAC/Linux)

```
# installation 
export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)"
echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install google-cloud-sdk
```

### 2.3 Setup 

```
gcloud init
# login 및 project setup 
```

# 2.4 환경 확인  
```
gcloud ml-engine models list
gcloud projects list

gcloud config set project [selected-project-id]
gcloud config set compute/zone [zone]
```


## 3. Example 

디렉토리 구조 
```
- pacakge
    - model.py (필요시)
    - simple_code.py
    - _init_.py
- setup.py
- ml_engine.sh
- config.yaml

```

###### simple_code.py

```python
%writefile ./package/simple_code.py  #jupyter로 코드 작성시 자동 저장 

import tensorflow as tf

const = tf.constant("hello tensorflow")

with tf.Session() as sess:
  result = sess.run(const)
  print(result)
```

###### ml_engine.sh
![](http://i.imgur.com/MXSlHjX.png)

```bash
#!/bin/bash

#### 파라미터 설정
JOB_NAME="task8"
PROJECT_ID=`gcloud config list project --format "value(core.project)"`
STAGING_BUCKET=gs://${PROJECT_ID}-ml
INPUT_PATH=${STAGING_BUCKET}/input
OUTPUT_PATH=${STAGING_BUCKET}/output/

PACKAGE_PATH='package.simple_code' # the directory of folder
MODULE_NAME='face_recog_model.model_localfile' # format: folder_name.source_file_name
SCALE_TIER='BASIC_GPU'

#### 실행 명령어 
gcloud ml-engine jobs submit training ${JOB_NAME} \
--runtime-version 1.2 # 텐서플로우 버젼 지정 가능 
--module-name=package.simple_code \
--package-path=$(pwd)/package \
--region=us-east1 \
--staging-bucket=$STAGING_BUCKET \
--scale-tier=$SCALE_TIER \
-- --input_dir="${INPUT_PATH}" \
-- --output_dir="${OUTPUT_PATH}" \


#### Multi_GPU
#--scale-tier=CUSTOM
#--config=./config.yaml


--train-files $TRAIN_DATA \  #TRAIN_DATA=gs://$BUCKET_NAME/data/adult.data.csv
--eval-files $EVAL_DATA \   #EVAL_DATA=gs://$BUCKET_NAME/data/adult.test.csv
--train-steps 1000 \
--verbosity DEBUG  \
--eval-steps 100
```

> [scale-tier옵션](https://cloud.google.com/ml-engine/docs/concepts/training-overview)

> Copy input.csv to Google Storage : `gsutil -m cp gs://cloudml-public/census/data/* data/`

###### __init__.py
- 일반 폴더가 아닌 패키지임을 표시하기 위해 사용
- 패키지를 초기화하는 파이썬 코드를 넣을 수 있다

###### setup.py
- 필요 패키지 설치 설정 : `pip`이용 

```
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['some_PyPI_package>=1.0']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My trainer application package.'
)

```



###### config.yaml
```
%writefile ./config.yaml
trainingInput:
  masterType: complex_model_m_gpu #GPU4개
  masterType: complex_model_l_gpu #GPU8개


````



## Commands

```shell
# 로컬에서 ml-engine을 이용한 모델 학습,예측 
# 올리기 전에 점검용 
gcloud ml-engine local train
gcloud ml-engine local prediction

# gclod 학습,예측 Job 제출 
gcloud ml-engine submit training
gcloud ml-engine submit prediction



# CloudML 결과 확인
gcloud ml-engine jobs stream-logs census_single_7

```




---
# Datalab 연동하여 실행하기 

1.Google Cloud shell 접속 

2.datalab 컴포넌트를 추가설치 : `gcloud components install datalab`


3.datalab create [instance name]
- 나중에  `datalab connect datalab-adioshun` 으로 다시 연결할 수 도있다 

> `gcloud config set compute/zone us-east1-c`

4. 삭제는 `datalab delete instance-name`

###### Error
```
Generating public/private rsa key pair.
Enter passphrase (empty for no passphrase): 

이런식으로 안넘어갈때가 있었는데 

{
 "error": {
  "errors": [
   {
    "domain": "global",
    "reason": "required",
    "message": "Login Required",
    "locationType": "header",
    "location": "Authorization"
   }
  ],
  "code": 401,
  "message": "Login Required"
 }
}
이걸 /home/mydiretory에다가 .sh 로 생성해서 다시 엔터치니까 그제서야 넘어갔다 
```

---

- [CLOUD MACHINE LEARNING ENGINE](https://cloud.google.com/ml-engine/)

 - [Cloud ML Engine Overview](https://cloud.google.com/ml-engine/docs/concepts/technical-overview)
 
 - [명령어 정리](https://cloud.google.com/sdk/gcloud/reference/ml-engine/)
 
- [CloudML 활용 예측](http://bcho.tistory.com/1180)

- [Tutorial: Training CIFAR-10 Estimator Model with TensorFlow 
for Hyperparameter Optimization](http://webofthink.tistory.com/76)





