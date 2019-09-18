# Tensorflow 저장 및 불러 오기 


## 1. 모델 zoo

- [Object Detection Model Zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md)

## 2. 저장 하기 

### 2,1 모델 체크포인트 .ckpt 
- 재학습 가능 모델에대한 메타정보 포함 
- 파일 크기가 크다 
- graph.pbtxt : 노드 정보가 모두 기록, .ckpt와 같이 생성 됨, input_graph 옵션의 입력값으로 활용됨 
- `tf.train.Saver().save(sess, 'trained.ckpt')` :학습한 변수 값들을  ckpt 체크포인트로 저장


### 2.2 pb 파일
- 재학습 불가능 
- 메타 데이타는 제외하고 모델과 가중치 값 포함 (모델의 그래프 + 학습된 변수값)
- tensorflow API를 이용한 C++ 프로그램에서 사용하는 포맷
- `tf.train.write_graph(sess.graph_def, ".", 'trained.pb', as_text=False)`: 그래프 저장


###### [참고] trained.ckpt+ trained.pb -> frozen_graph.pb 변환 툴 : [freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py)
- out_node_names 옵션 : 자기가 사용하고자 하는 모델의 출력 노드 지정, `graph.pbtxt`파일 참고 
- 오류시 파일 상단 설정 부분에 `--input_binary=true` 추가 


> 참고 : [The tensorgraph is a example show how to generate, load graph from tensorflow](https://github.com/JackyTung/tensorgraph)



