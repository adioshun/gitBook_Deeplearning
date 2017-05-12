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

# Methods 

We immediately found that different sampling techniques do not allow for a direct comparison of resource utilisation.
> Different  sampling techniques으로는 서로 비교가 불가능 하다 

For example, central-crop (top-5 validation) errors of a single run of VGG-16 (Simonyan & Zisserman, 2014) and GoogLeNet (Szegedy et al., 2014) are 8.70% and 10.07% respectively, revealing that VGG-16 performs better than GoogLeNet. 


When models are run with 10-crop sampling, 2 then the errors become 9.33% and 9.15% respectively, and therefore VGG-16 will perform worse than GoogLeNet, using a single central-crop. 

For this reason,we decided to base our analysis on re-evaluations of top-1 accuracies 3 for all networks with a single central-crop sampling technique (Zagoruyko, 2016).

For inference time and memory usage measurements we have used Torch7 (Collobert et al., 2011)with cuDNN-v5 (Chetlur et al., 2014) and CUDA-v8 back-end. 
> `예측 시간`과 `메모리 사용량` 측정을 위해 cuDNN기반의 Torch7를 이용하였다. 

Operation counts were obtained using an open-source tool that we developed (Paszke, 2016). 
> `Operation counts`는 자체 개발한 툴을 사용하였다. 

For measuring the power consumption, a Key sight 1146B Hall effect current probe has been used with a Key sight MSO-X 2024A 200 MHz digital oscilloscope with a sampling period of 2 s and 50 kSa/s sample rate. 
> `전력 사용량` 측정을 위해서는 ....

The system was powered by a Key sight E3645AGPIB controlled DC power supply.

# Result 
1~8개 항목에 대한 결과 그래프적 표현 (원문 참고)

# 4 CONCLUSIONS

