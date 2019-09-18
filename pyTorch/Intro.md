# PyTorch YOLO2

> [PyTorch 튜토리얼 1~10]()
> [PyTorch MNIST Example similar to TensorFlow Tutorial](https://github.com/rickiepark/pytorch-examples/blob/master/mnist.ipynb)


## 1. 개요 

## 1.1 기본 import 패키지 
```python 
import torch # arrays on GPU
import torch.autograd as autograd #build a computational graph
import torch.nn as nn ## neural net library
import torch.nn.functional as F ## most non-linearities are here
import torch.optim as optim # optimization package
```


### 1.2 파일 입력 : `DataLoader()`

```
train_loader = torch.utils.data.DataLoader()
```

## 2. Modeling

#### A. 정의된 모델 불러 오기 

```python
cuda = torch.cuda.is_available()
start_epoch = 0
start_iteration = 0
resume = False

model = torchfcn.models.FCN8s(n_class=21)    #import torchfcn 하였기 떄문에 가능

# 이전에 저장되었던 부분 부터 하기 
if resume:
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    start_iteration = checkpoint['iteration']
else:
    fcn16s = torchfcn.models.FCN16s()
    fcn16s.load_state_dict(torch.load(torchfcn.models.FCN16s.download())) 
    model.copy_params_from_fcn16s(fcn16s)  #파라미터 복사 
if cuda:
    model = model.cuda()
```


#### B. 직접 정의하기 
```python 
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        # input is 28x28
        # padding=2 for same padding
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        # feature map size is 14*14 by pooling
        # padding=2 for same padding
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        # feature map size is 7*7 by pooling
        self.fc1 = nn.Linear(64*7*7, 1024)
        self.fc2 = nn.Linear(1024, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64*7*7)   # reshape Variable
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
model = MnistModel()
```

###### [Tip] 모델내 파라미터(Weight, biase)확인 
```python
for p in model.parameters():
    print(p.size())

```

```python
params = list(model.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight


출처: http://bob3rdnewbie.tistory.com/316 [Newbie Hacker]
```

## 3. Training : `model.train()`

### 3.1 학습 알고리즘 정의 

```python 
optimizer = optim.Adam(model.parameters(), lr=0.0001)
```

### 3.2 Trainer 정의 

```python
trainer = torchfcn.Trainer(
    cuda=cuda,
    model=model,
    optimizer=optim,
    train_loader=train_loader,
    val_loader=val_loader,
    out='./log/',
    #max_iter=cfg['max_iteration'],
    max_iter=100000,
    #interval_validate=cfg.get('interval_validate', len(train_loader)),
    interval_validate=4000,

)
trainer.epoch = start_epoch
trainer.iteration = start_iteration
```

최종 학습 수행 명령어 : `trainer.train()`


###### [참고] Loss 함수 정의 : `nn.MSELoss()`
```python 
output = net(input)
target = Variable(torch.arange(1, 11))  # a dummy target, for example
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
```

###### [참고] 역전파 : ` loss.backward()`

loss.backward()를 호출하고 backward() 호출 이전과 이후의 바이어스 그라디언트를 살펴볼 것이다.
```python
net.zero_grad()     # zeroes the gradient buffers of all parameters
print(net.conv1.bias.grad)

loss.backward()

print(net.conv1.bias.grad)
```




###### [전체 코드] 

```python 
model.train()
train_loss = []
train_accu = []
i = 0
for epoch in range(15):
    for data, target in train_loader:
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()    # calc gradients
        train_loss.append(loss.data[0])
        optimizer.step()   # update gradients
        prediction = output.data.max(1)[1]   # first column has actual prob.
        accuracy = prediction.eq(target.data).sum()/batch_size*100
        train_accu.append(accuracy)
        if i % 1000 == 0:
            print('Train Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(i, loss.data[0], accuracy))
        i += 1
```



model.train()

## 4. Testing : `model.eval()`

model.eval() 



###### [전체 코드] 

```python
model.eval()
correct = 0
for data, target in test_loader:
    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).sum()

print('Test set: Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))
```
## 5. Fine tuning  


---

|제목|YOLO2|
|-|-|
|코드|[marvis](https://github.com/marvis/pytorch-yolo2)|
|참고||

# PyTorch YOLO2

## 1. 개요 

## 2. 설치 

```bash
# pytorch 설치 
conda create -n py2torch python=2.7 ipykernel
source activate py2torch
conda install pytorch torchvision cuda80 -c soumith

#YOLO설치
git clone git@github.com:marvis/pytorch-yolo2.git
wget http://pjreddie.com/media/files/yolo.weights
python detect.py cfg/yolo.cfg yolo.weights data/dog.jpg
```

## 3. Training

#### Training YOLO on VOC
##### Get The Pascal VOC Data
```
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
```
##### Generate Labels for VOC
```
wget http://pjreddie.com/media/files/voc_label.py
python voc_label.py
cat 2007_train.txt 2007_val.txt 2012_*.txt > voc_train.txt
```
##### Modify Cfg for Pascal Data
Change the cfg/voc.data config file
```
train  = train.txt
valid  = 2007_test.txt
names = data/voc.names
backup = backup
```
##### Download Pretrained Convolutional Weights
Download weights from the convolutional layers
```
wget http://pjreddie.com/media/files/darknet19_448.conv.23
```
or run the following command:
```
python partial.py cfg/darknet19_448.cfg darknet19_448.weights darknet19_448.conv.23 23
```
##### Train The Model
```
python train.py cfg/voc.data cfg/yolo-voc.cfg darknet19_448.conv.23
```
##### Evaluate The Model
```
python valid.py cfg/voc.data cfg/yolo-voc.cfg yolo-voc.weights
python scripts/voc_eval.py results/comp4_det_test_
```
mAP test on released models
```
yolo-voc.weights 544 0.7682 (paper: 78.6)
yolo-voc.weights 416 0.7513 (paper: 76.8)
tiny-yolo-voc.weights 416 0.5410 (paper: 57.1)

```
## 4. Testing 


## 5. Fine tuning  





--- 
변경 사항 
```python 
# detect.py

PATH_TO_TEST_IMAGES_DIR = '/home/hjlim99/test_images/'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'frame{}.jpg'.format(i)) for i in range(1, 11561) ]


def detect(cfgfile, weightfile, imgfile):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    for image_path in TEST_IMAGE_PATHS:
        img = Image.open(image_path).convert('RGB')
        sized = img.resize((m.width, m.height))
    
    
    #img = Image.open(imgfile).convert('RGB')
    #sized = img.resize((m.width, m.height))
    
        for i in range(2):
            start = time.time()
            boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
            finish = time.time()
            if i == 1:
                print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

        class_names = load_class_names(namesfile)
        #plot_boxes(img, boxes, 'predictions.jpg', class_names)
        filename = image_path.replace('/home/hjlim99/test_images/', '')
        plot_boxes(img, boxes, './result/{}'.format(filename), class_names)
        #'./save/{}.png'.format(image_path)


cfgfile = "cfg/yolo.cfg"
weightfile = "weight/yolo.weights"
imgfile = "data/dog.jpg"

detect(cfgfile, weightfile, imgfile)
```

