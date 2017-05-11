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

It is also apparent(명백한) that **multiplying the number of paths through the network** is a recent trend that is illustrated in the progression from Alexnet to Inception to ResNets. 
        - For example, Veit et al. (2016)show that ResNets can be considered to be an exponential ensemble of networks with different lengths. 

> 최근 트렌드는 **multiplying the number of paths**    



|Design Pattern 2|Proliferate Paths is based on the idea that ResNets can be an exponentialensemble of networks with different lengths. |
|-|-|



One proliferates paths by including a multiplicity ofbranches in the architecture. 
Recent examples include FractalNet (Larsson et al. 2016), Xception(Chollet 2016), and Decision Forest Convolutional Networks (Ioannou et al. 2016). 
Scientists have embraced simplicity/parsimony for centuries. 
Simplicity was exemplified in thepaper ”Striving for Simplicity” (Springenberg et al. 2014) by achieving state-of-the-art results withfewer types of units. 


Design Pattern 3: Strive for Simplicity suggests using fewer types of unitsand keeping the network as simple as possible. 

We also noted a special degree of elegance inthe FractalNet (Larsson et al. 2016) design, which we attributed to the symmetry of its structure.

Design Pattern 4: Increase Symmetry is derived from the fact that architectural symmetry is typicallyconsidered a sign of beauty and quality. 

In addition to its symmetry, FractalNets also adheres to theProliferate Paths design pattern so we used it as the baseline of our experiments in Section 4.
























---



<a name="Alexander">[1]</a> Christopher Alexander. The timeless way of building, volume 1. New York: Oxford University Press, 1979  <br/>

<a name="Gamma">[2]</a> Erich Gamma, Richard Helm, Ralph Johnson, and John Vlissides. Design patterns: elements of reusable object-oriented software. Pearson Education India, 1995 <br/>


<a name="Alexander">[1]</a> Christopher Alexander. The timeless way of building, volume 1. New York: Oxford University Press, 1979  <br/>