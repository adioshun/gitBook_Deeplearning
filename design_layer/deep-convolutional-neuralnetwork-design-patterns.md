# Deep Convolutional Neural Network Design Patterns 

> Abs. 최근 많은 네트워크들이 개발되고 있지만 이를 사용하는 활용자의 관련 지식 미흡으로 개발과 적용의 기술적 갭이 존재 한다. 본 논문에서는 갭 해결을 위한 정보 제공과 몇가지 새 네트워크를 제안 하려 합니다. [[Github]](https://github.com/iPhysicist/CNNDesignPatterns)

# 1. INTRODUCTION

본 논문에서는 14개의 Original Design Pattern에 대하여 살펴 볼것이다. 


## 기존 연구 
- Alexander (1979) in regards to the architectures of buildings and towns[[1]](#Alexander)
    - The basis of design patterns is that they resolve a conflict of forces in a given context and lead to an equilibrium analogous to the ecological balance in nature. 
    - Design patterns are both highly specific, making them clear to follow, and flexible so they can be adapted to different environments and situations.
    

- Gamma et al. (1995) the “gang of four” applied the concept of
design patterns to the architecture of object-oriented software[[2]](#Gamma)
    - This classic computer science book describes **23 patterns** that resolve issues prevalent in software design, such as “requirements always change”. 


> 위 2개의 기존 연구를 기반으로 본 연구는 진행 되었다. 모든 네트워크의 패턴을 다 살펴 보기는 어려우니 본 논문에서는 **"CNN"**의 **"Classification"**에 초점을 맞추었다. 


|Design Pattern 1|Architectural Structure follows the Application|
|-|-|

# 2 RELATED WORK

- ”Neural Networks: Tricks of the Trade” (Orr & Müller, 2003) **contains recommendations for network models** but without reference to the vast amount of research in the past few years

- Szegedy et al. (2015b) where those authors describe a few design principles based on their experiences.
> 본 논문과 가장 유사한 내용을 포함하고 있음 


we focus on recent innovations in convolutional neural networks architectures and, in particular, on **Residual Networks**(He et al., 2015) and its recent family of variants. 

- We start with Network In Networks (Lin et al.,2013), which describes a hierarchical network with a small network design repeatedly embedded in the overall architecture. 

- Szegedy et al. (2015a) incorporated this idea into their Inception ar-chitecture. 
    - Later, these authors proposed modifications to the original Inception design (Szegedyet al., 2015b). 


- A similar concept was contained in the multi-scale convolution architecture (Liao &Carneiro, 2015). 

- In the meantime, Batch Normalization (Ioffe & Szegedy, 2015) was presented as a unit within the network that makes training faster and easier.


- Before the introduction of Residual Networks, a few papers suggested skip connections. 
    - Skip connections were proposed by Raiko et al. (2012). 
    - Highway Networks (Srivastava et al., 2015) use a **gating mechanism** to decide whether to combine the input with the layer’s output and showed how these networks allowed the training of very deep networks. 
    - The DropIn technique (Smith et al.,2015; 2016) also trains very deep networks by allowing a layer’s input to skip the layer. 


- The concept of stochastic depth via a drop-path method was introduced by Huang et al. (2016b).


- Residual Networks were introduced by He et al. (2015), where the authors describe their network that won the 2015 ImageNet Challenge. 
    - They were able to extend the depth of a network from tensto hundreds of layers and in doing so, improve the network’s performance. 
    - The authors followed up with another paper (He et al., 2016) where they investigate why identity mappings help and report results for a network with more than a thousand layers. 
    


- The research community took notice of this architecture and many modifications to the original design were soon proposed.

- The Inception-v4 paper (Szegedy et al., 2016) describes the impact of residual connections on their Inception architecture and compared these results with the results from an updated Inception design.


- The Resnet in Resnet paper (Targ et al., 2016) suggests a duel stream architecture. 
    -Veit et al. (2016)provided an understanding of Residual Networks as an ensemble of relatively shallow networks.
    - These authors illustrated how these residual connections allow the input to follow an exponential number of paths through the architecture. 
    

- At the same time, the FractalNet paper (Larsson et al.,2016) demonstrated training deep networks with a symmetrically repeating architectural pattern.

- As described later, we found the symmetry introduced in their paper intriguing. 

- In a similar vein,Convolutional Neural Fabrics (Saxena & Verbeek, 2016) introduces a three dimensional network,where the usual depth through the network is the first dimension.

- Wide Residual Networks (Zagoruyko & Komodakis, 2016) demonstrate that simultaneously increasing both depth and width leads to improved performance. 

- In Swapout (Singh et al., 2016), each layer can be dropped, skipped, used normally, or combined with a residual. 

- Deeply Fused Nets (Wanget al., 2016) proposes networks with multiple paths. 

- In the Weighted Residual Networks paper (Shen& Zeng, 2016), the authors recommend a weighting factor for the output from the convolutional layers, which gradually introduces the trainable layers. 

- Convolutional Residual Memory Networks(Moniz & Pal, 2016) proposes an architecture that combines a convolutional Residual Network with an LSTM memory mechanism. 

- For Residual of Residual Networks (Zhang et al., 2016), the authors propose adding a hierarchy of skip connections where the input can skip a layer, a module, or any number of modules. 

- DenseNets (Huang et al., 2016a) introduces a network where each module isdensely connected; that is, the output from a layer is input to all of the other layers in the module. 

- In the Multi-Residual paper (Abdi & Nahavandi, 2016), the authors propose expanding a residual blockwidth-wise to contain multiple convolutional paths. 

> Appendix A에서 위 내용들을 정리 하였다. 


# 3 DESIGN PATTERNS

## 3.1 HIGH LEVEL ARCHITECTURE DESIGN

|Design Pattern 2|Proliferate(증식,확산) Paths is based on the idea that ResNets can be an exponential ensemble of networks with different lengths. |
|-|-|


It is also apparent(명백한) that **multiplying the number of paths through the network** is a recent trend that is illustrated in the progression from Alexnet to Inception to ResNets. 
        - For example, Veit et al. (2016)show that ResNets can be considered to be an exponential ensemble of networks with different lengths. 

> 최근 트렌드는 **multiplying the number of paths** 이다.  

예" FractalNet (Larsson et al. 2016), Xception(Chollet 2016), and Decision Forest Convolutional Networks (Ioannou et al. 2016). 


|Design Pattern 3|**Strive(분투하다) for Simplicity** suggests using fewer types of units and keeping the network as simple as possible. |
|-|-|

Simplicity was exemplified(대표적 예) in the paper ”Striving for Simplicity” (Springenberg et al. 2014) by achieving state-of-the-art results with fewer types of units. 


|Design Pattern 4|Increase Symmetry(대칭) is derived from the fact that architectural symmetry is typically considered a sign of beauty and quality.|
|-|-|

We also noted a special degree of elegance in the FractalNet (Larsson et al. 2016) design, which we attributed to the symmetry of its structure.


In addition to its symmetry, FractalNets also adheres to the Proliferate Paths design pattern. 
> 간결성 이외에도 **FractalNets**는 Proliferate Paths design pattern도 포함 하고 있다. 


An essential element of design patterns is the examination of trade-offs in an effort to understand the relevant forces. 
> 디자인 패턴의 중요 요소중 하나는 examination of **trade-offs** in an effort to understand the relevant forces. 

One fundamental trade-off is the maximization of representational power versus elimination of redundant and non-discriminating information. 
> 트레이드 오프 1 : 표현력 극대화(maximization of representational power) Vs. 비 식별 정보 제거 (elimination of non-discriminating information.)


|Design Pattern 5|**Pyramid Shape** says there should be an overall smooth down sampling combined with an increase in the number of channels throughout the architecture.|
|-|-|

It is universal in all convolutional neural networks that the activations are downsampled and the number of channels increased from the input to the final layer, which is exemplified in Deep Pyramidal Residual Networks (Han et al. (2016)). 
> CNN에서 일반적으로 마지막 레이어 전에는 `Activation은 다운샘플링` 하고 `Channel의 수는 증가` 시킨다. 이것은  `Deep Pyramidal Residual Networks`에 잘 나타나 있다. 

Another trade-off in deep learning is training accuracy versus the ability of the network to generalize to non-seen cases. 
> 트레이드 오프 2: 학습 정확도(training accuracy) Vs. 일반화(Generalize to non-seen cases)

Regularization is commonly used to improve generalization, which includes methods such as `dropout`(Srivastava et al. 2014a) and `drop-path` (Huang et al. 2016b). 
We believe regularization techniques and prudent noise injection during training improves generalization (Srivastava et al.2014b, Gulcehre et al. 2016). 
> dropout(일부로 Noise추가)과 drop-path같은 Regularization 기법들은 모델의 일반화를 증가 시킨다. 

|Design Pattern 6|**Over-train** includes any training method where the network is trained on a harder problem than necessary to improve generalization performance of inference. |
|-|-|



|Design Pattern 7|**Cover the Problem Space** with the training data is another way to improve generalization|
|-|-|


e.g., Ratner et al. 2016, Hu et al. 2016, Wong et al. 2016, Johnson-Robersonet al. 2016). 

Related to regularization methods, cover the problem space includes the use of noise(Rasmus et al. 2015, Krause et al. 2015, Pezeshki et al. 2015) and data augmentation, such as randomcropping, flipping, and varying brightness, contrast, and the like.
> `cover the problem space`는 Noise와 Data augmentation을 사용한다. 

## 3.2 DETAILED ARCHITECTURE DESIGN

A common thread throughout many of the more successful architectures is to make each layer’s “job”  easier.   
> 대부분의 좋은 성능을 보이는 네트워크들은 각 Layer의 'Job'을 Easier하게 하였다. 

Use  of  very  deep  networks  is  an  example  because  any  single  layer  only  needs  to incrementally modify the input, and this partially explains the success of Residual Networks, since in very deep networks, a layer’s output is likely similar to the input; hence adding the input to the layer’s output makes the layer’s job incremental. 

Also, this concept is part of the motivation behind design pattern 2 but it extends beyond that. 
> 이 컨셉은 Pattern 2의 연장선에 있다. 

### each layer’s “job”  easier하는 첫번째 방법

|Design Pattern 8|**Incremental Feature Construction** recommends using short skip lengths in ResNets.| 짧은 Skip Length사용할것을 추천|
|-|-|-|

A recent paper (Alain & Bengio (2016)) showed in an experiment that using an identity skip length of 64 in a network of depth 128 led to the first portion of the network not being trained. 
> 최근 연구에 따르면 128층의 레이어에서 skip length 64를 쓰는것은 레이어의 첫번째 위치한 것은 학습이 일어 나지 않았다. 

### each layer’s “job”  easier하는 두번째 방법

|Design Pattern 9|**Normalize Layer Inputs** is another way to make a layer’s job easier|
|-|-|

Normalization of layer inputs has been shown to improve training and accuracy but the underlying reasons are not clear (Ioffe & Szegedy 2015, Ba et al. 2016, Salimans & Kingma 2016). The Batch Normalization paper (Ioffe & Szegedy 2015) attributes the benefits to handling internal covariate shift, while the authors of streaming normalization (Liao et al. 2016) express that it might be otherwise.  We feel that normalization puts all the layer’s input samples on more equal footing (analogous to a units conversion scaling), which allows back-propagation to train more effectively. 

















---



<a name="Alexander">[1]</a> Christopher Alexander. The timeless way of building, volume 1. New York: Oxford University Press, 1979  <br/>

<a name="Gamma">[2]</a> Erich Gamma, Richard Helm, Ralph Johnson, and John Vlissides. Design patterns: elements of reusable object-oriented software. Pearson Education India, 1995 <br/>


<a name="Alexander">[1]</a> Christopher Alexander. The timeless way of building, volume 1. New York: Oxford University Press, 1979  <br/>