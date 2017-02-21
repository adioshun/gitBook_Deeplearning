# 출처
*2017.02.21 Github/gitBook Sync
*2017.02.21 second sync

## 모두를 위한 딥러닝\(시즌1\)

* [Youtube](https://www.youtube.com/watch?v=BS6O0zOGX4E&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm)
* [강의 웹사이트](http://hunkim.github.io/ml/)
* [질문사이트](http://qna.iamprogrammer.io/c/dev/ml)
* [요약정리](http://pythonkim.tistory.com/notice/25)

## 텐서플로우 첫걸음

* 조르디 토레스 지음, 박해선 옮김, 한빛미디어 출판

## 딥러닝의 역사

* 2009 : 음성 인식
* 2012 : 컴퓨터 비젼
* 2014 : 번역 

## 텐서플로우

* [파이썬\_킴 블로그](http://pythonkim.tistory.com/category/%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C%EC%9A%B0)

# 딥러닝의 역사

1. 1950년대 :뇌의 뉴런\(단층\)을 살펴 보니 간단\(Wx+B\)하고 이를 이용하여 AND / OR등\(당시 중요 평가지표\)의 문제 해결이 가능함 확인 
   * XOR도 중요한 지표인데 기존\(단층\)뉴런으로는 해결 할수 없다. 
   * 암흑기
2. 1969년 :  다층 뉴런을 사용하면 XOR도 해결 가능하다 by Minsky, MIT
   * 단, 다층 뉴런간의 W,B학습이 불가능 하다. 
   * No one on the earth had found a viable way to train
   * 10~20년 1차 암흑기
3. 1974,1982년 : Backpropagation 알고리즘 해결 가능 by Paul Werbos
   * 1986년 Hilton교수에 의하여 재발견 및 알려짐 
4. 1980년 : CNN알고리즘을 이용하면 이미지 인식에 큰 발전 가능 by LeCun
   * \[??\] 86년에 Backpropagation이 알려졌는데 그럼 cNN은 backpropagatio이 없나?
5. 1995년 Backpropagation은 2~3개의 레이어는 가능하지만, 깊은 층에서는 학습이 잘 되지 않음
   * SVM, RF가 더 좋은 결과 보임 
   * Vanishing Gradient
   * 2차 암흑기 (1986-2006)
6. 2006년 깊은층의 학습을 위해서 아래의 내용들을 해결 하면 가능하다. by hilton, Bengio
   * Our labeled datasets ware thousands of times too small
   * Our computers were million of times too slow
   * We initialized the weight in a stupid way (RBM, DDeep Belief Network)
   * We used the wrong type of non-Linearity (Sigmoid -> Relu로 해결)
   > Deep Learning으로 재 명명 \(뉴럴넷이 부정적 이미지가 많음\)
7. 2012년 ImageNet 경연대회에서 딥러닝기반 AlexNet으로 26.2% -&gt; 15,3%로 떨어트림 by Hilton교수랩 Alex박사 과정 
8. 2015년 인간에러 5% 보다 좋은 성능 보임



