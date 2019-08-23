# Neural Architecture Search for Object Detection in Point Cloud


# [Neural Architecture Search for Object Detection in Point Cloud](https://medium.com/seoul-robotics/neural-architecture-search-for-object-detection-in-point-cloud-f2d57a5953d5)


# State of the Art

NAS를 위한 3가지 방법론 `Fundamentally there are three different approaches for the Search Strategy, i.e. Reinforcement Learning (RL), Evolutionary Methods and Gradient-Based methods.`

- **Reinforcement Learning** (e.g. [Zoph et al., 2018](https://arxiv.org/abs/1707.07012)) frames the problem with an agent that aims to find a suitable architecture in order to maximize its reward which is the network performance.
- **Evolutionary Methods** (e.g. [Real et al. 2019](https://arxiv.org/abs/1802.01548)) make use of genetic algorithms to sample a _parent_ from a population of networks, which is used to produce _offspring_ models by applying mutations to the architecture, for instance, this can be the change connections, operations or similar.
- **Gradient-Based** (e.g. [Liu et al. 2018](http://arxiv.org/abs/1806.09055) ) methods use _continuous relaxation_ to make use of gradient descent methods for the network architecture optimization. Instead of fixing an architecture this approach uses a convex combination of multiple architectures.

처음 두개는 계산 부하가 큰 단점이 있다. `The major downside for Reinforcement Learning and Evolutionary Methods is that they both tend to be **computationally expensive** with the search needing as much as **2,000 and 3,150 GPU days** respectively.`
- 반면 장점은 :  On the other hand, those methods can be used for multi-objective optimization by adopting the optimized metric accordingly and are therefore more flexible.


랜덤 샘플링을 이용한 방법도 좋은 결과를 보였음  **A recent critic** on the above approaches was stated by [Liam et al. (2019)](https://arxiv.org/abs/1902.07638). They achieved the same performance as the leading approaches by **randomly sampling** the network architectures. Liam et al. therefore argue that the correctly chosen restricted architecture search space that most NAS methods require is the biggest reason for the good performance of the recent work in the field.

# Neural Architecture Search for Object Detection in Point Cloud Data


대부분의  NAS는 이미지에 초점을 두고 있으며 포인트 클라우드 관련은 몇가지 없다. `In contrast to that, most of the work in NAS was conducted in **image classification** (e.g. [Zoph et al., 2018](https://arxiv.org/abs/1707.07012), [Real et al. 2019](https://arxiv.org/abs/1802.01548)) and only a hand full in **object detection** ([Tan et al. 2018](https://arxiv.org/abs/1807.11626), [Liu et al. 2019](http://arxiv.org/abs/1901.02985)). To the best of our knowledge, there has not been any work published in the realm of NAS for **object detection in point cloud data**. Which makes it both a challenging and promising problem to solve.`


계산 부하를 줄이기 위해 Auto-DeepLab에서 개발한 gradient기반 방식을 채용 하였다. `In order to **avoid the immense computational cost** of Reinforcement Learning and Evolutionary Methods, I aimed to extend a recent **gradient-based** approach by [Liu et al. 2019](http://arxiv.org/abs/1901.02985) dubbed **Auto-DeepLab.**`

Auto-DeepLab의 방식은 이미지를 대상으로 하는것이기에 PIXOR의 3D 처리 방식과 혼합 하였다. `As Auto-DeepLab was developed for image segmentation it does not directly translate to point cloud data. I, therefore, mixed in some of the manually found architecture elements and other methods for object detection in point cloud data reported by [Yang et al. (2018)](https://arxiv.org/abs/1902.06326) in a paper called **PIXOR.**`

## Methods
