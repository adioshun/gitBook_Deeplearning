# pyTorch Tips 



## Multi-GPU

- [PyTorch Multi-GPU 제대로 학습하기](https://medium.com/daangn/pytorch-multi-gpu-%ED%95%99%EC%8A%B5-%EC%A0%9C%EB%8C%80%EB%A1%9C-%ED%95%98%EA%B8%B0-27270617936b)



- [Multi GPU](http://bob3rdnewbie.tistory.com/321?category=780658)


## Parallel Computing

http://bob3rdnewbie.tistory.com/346?category=780658


## PyTorch — Dynamic Batching

TensorFlow Fold를 참고한 PyTorch 의 Dynamic Batching 구현인 [TorchFold](https://medium.com/@ilblackdragon/pytorch-dynamic-batching-f4df3dbe09ef)[(Code)](https://github.com/nearai/pytorch-tools/blob/master/pytorch_tools/torchfold.py)


## PyTorch + einops

Einstein notation 은 복잡한 텐서 연산을 표기하는 방법입니다. 이름이 생소할 수는 있어도 사실 선형대수학을 비롯해서 벡터/행렬 등을 표기할 때 일반적으로 쓰는 방법이죠.
https://en.wikipedia.org/wiki/Einstein_notation

딥러닝에서 쓰이는 많은 연산은 Einstein notation 으로 쉽게 표기할 수 있습니다. 기존에도 프레임워크마다 'einsum' 라는 API 가 있긴 했지만 각각 작성법이 다르고 기능이 제한적이었는데, 마침 반갑게도 얼마 전에 numpy, pytorch, tensorflow 등 여러 프레임워크를 동시에 지원하는 einops (https://github.com/arogozhnikov/einops) 라는 라이브러리가 공개되었습니다.

그리고 어제 einops 에서 einops 의 'Rearrange / Reduce' API 를 이용해서 pytorch 코드를 어떻게 더 간단히 작성할 수 있는지 샘플 코드들을 공개했네요!
Convolution, Pixel Shuffler, Gram Matrix, Channel Shuffle, RNN, CBHG, Attention, Transformer, Glow, YOLO 등 다양한 예제들이 있습니다.
코드 라인 수가 상당히 줄고 훨씬 직관적으로 보입니다.
그럼에도 불구하고 속도는 순수 pytorch 코드와 차이가 없다고 하는군요 :)

* Sample codes: https://arogozhnikov.github.io/einops/pytorch-examples.html

* Tutorial: https://github.com/arogozhnikov/einops/tree/master/docs





