VGGNet



ILSVRC 2014년 대회에서 2등을 한 구조로

영국 옥스포트 대학교의 Karen Simonyan과 Andrew Zisserman에 의해서 개발이 되었다.

대량의 이미지를 인식함에 있어, 망의 깊이(depth)가 정확도에 어떤 영향을 주는지를 보여주었다.

망의 시작부터 끝까지 동일하게 3x3 convolution과 2x2 max pooling을 사용하는 단순한(homogeneous) 구조에서

depth가 16일 때 최적의 결과가 나오는 것을 보여줬다.


GoogLeNet에 비해 분류 성능은 약간 떨어졌지만, 다중 전달 학습 과제에서는 오히려 더 좋은 결과가 나왔다.

그러므로 CNN 연구 그룹에서는 VGGNet의 구조를 좀 더 선호하는 경향이 있다.

단점이라면, 메모리 수와 파라미터의 수가 크다는 점이다.