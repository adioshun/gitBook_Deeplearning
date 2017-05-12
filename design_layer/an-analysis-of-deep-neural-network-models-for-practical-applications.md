# An Analysis of Deep Neural Network Models for Practical Applications

we present a comprehensive analysis of important metrics in practical applications: 
- accuracy, memory footprint, parameters, operations count, inference time, power consumption. 

> 실질적 applications에 대한 중요 Metrics 분석 수행 

Key findings are(주요 시사점): 
- (1) power consumption is independent of batch size and architecture; 
- (2) accuracy and inference time are in a hyperbolic relationship; 
- (3) energy constraint is an upper bound on the maximum achievable accuracy and model complexity; 
- (4) the number of operations is a reliable estimate of the inference time. 

> 전력 소비량은 배치사이즈나 아키텍쳐와 상관 없다, 정확도와 예측시간은 hyperbolic 관계이다. 전력사용량...., Operation 수....

We believe our analysis provides a compelling set of information that helps design and engineer efficient DNNs.

# 1. Introoduction 

## 기존 ImageNet 경진대회 문제점 
- Firstly, it is now normal practice to run several trainedinstances of a given model over multiple similar instances of each validation image. 
    - This practice,also know as model averaging or ensemble of DNNs, dramatically increases the amount of com-putation required at inference time to achieve the published accuracy. 

- Secondly, model selection is hindered by the fact that different submissions are evaluating their (ensemble of) models a differentnumber of times on the validation images, and therefore the reported accuracy is biased on the spe-cific sampling technique (and ensemble size). 

- Thirdly, there is currently no incentive in speeding up inference time, which is a key element in practical applications of these models, and affects resource utilisation, power-consumption, and latency.


> 본 논문에서는 ImageNet에 최근 4년간 참여한 알고리즘들을 computational requirements & accuracy의 측면에서 살펴 보려 한다. 

