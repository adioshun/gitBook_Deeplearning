# Convolutional Neural Networks

## 1. 기본 동작 
###### Step 1. 입력 이미지, 필터 정의 

![](http://i.imgur.com/rmuzh48.png)

- 입력 : 이미지 32x32x3
- 필터 : 5x5x3 (6개 필터)
- 연산 : Convolve = slide over the image spatially, computing dot products

###### Step 2. Convolve연산 개요

![](http://i.imgur.com/RQtrq6e.png)


Convolve연산 방법 : 필터 5x5x3(w)$$\cdot$$ 이미지의 5x5x3영역(x) + b
Convolve연산 목표 : 1개의 값 출력

###### Step 3. Convolve연산 수행 (6개중 1개의 필터)

![](http://i.imgur.com/h2zRBF2.png)

이미지의 모든 영역에 Convolve작업을 하여 28x28x1의 Activation Map생성 


###### Step 4. Convolve연산 수행 (6개중 6개의 필터)

![](http://i.imgur.com/RLVvB83.png)

6개의 필터(5x5x3)을 모두 Convolve연산 수행하여 6개의 Activation Map(28x28x1) 도출

- 출력 : New Conv image (28x28x6) 

###### Step 5. Convolutional Layers 반복 수행
ConvNet is a sequence of Convolutional Layers, interspersed with activation functions
> Step 1으로 돌아감 

![](http://i.imgur.com/54WyhWs.png)

- 입력 : New Conv image  28x28x**6**
- 필터 : 5x5x**6** (10개 필터)
    - (6:입력레이어의 D가 6이므로) 
- 연산 : Convolve = slide over the image spatially, 
- 출력 : New Conv image 24x24x10 (10은 필터수)

> Notice that : 동일한 필터크기(5x5)를 적용해도 결과물 크기가 32 - 28 - 24로 **급격히** 줄어듬shrink (깊은 레이어 생성 불가) $$\Rightarrow$$ Stride/pad개념으로 해결 가능 


## 2. Advanced 동작 (레이어 shrink 문제 해결

### 2.1 Stride 

필터가 이미지를 이동시 간격

### 2.2 pad 

테두리에 0으로 된 값 입력 


### 2.3 레이어 계산 방법 (Stride  + pad 적용시)

![](http://i.imgur.com/vKLwri1.png)

입력 : 32x32x3
필터 : 10개의 5x5필터, Stide 1, pad 2

결과 : 
- (입력이미지 크기 + 양면(2) x pad-필터 크기)/Strid + 1 
- (32 + 2 x 2 - 5)/1 + = 32 $$\Rightarrow$$ 32x32x필터갯수



### 2.4 파라미터  수 계산 방법 (Stride  + pad 적용시)

![](http://i.imgur.com/L2VBOva.png)



## 3. Summary 

![](http://i.imgur.com/cxadXWV.png)

###### Step 