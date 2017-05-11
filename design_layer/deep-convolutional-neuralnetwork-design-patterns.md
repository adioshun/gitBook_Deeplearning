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


# 2 RELATED WORK

- ”Neural Networks: Tricks of the Trade” (Orr & Müller, 2003) **contains recommendations for network models** but without reference to the vast amount of research in the past few years

- Szegedy et al. (2015b) where those authors describe a few design principles based on their experiences.
> 본 논문과 가장 유사한 내용을 포함하고 있음 



---



<a name="Alexander">[1]</a> Christopher Alexander. The timeless way of building, volume 1. New York: Oxford University Press, 1979  <br/>

<a name="Gamma">[2]</a> Erich Gamma, Richard Helm, Ralph Johnson, and John Vlissides. Design patterns: elements of reusable object-oriented software. Pearson Education India, 1995 <br/>


<a name="Alexander">[1]</a> Christopher Alexander. The timeless way of building, volume 1. New York: Oxford University Press, 1979  <br/>