# Rethinking the Inception Architecture for Computer Vision

논문의 2장만 정리 

## 2. General Design Principles

### Avoid representational bottlenecks

Avoid representational bottlenecks, especially early in the network. 
> `representational bottlenecks` 제거하기. 특히 네트워크 앞단에 있는거는 중요 

Feed-forward networks can be represented by an acyclic graph from the input layer(s) to the classifier or regressor. This defines a clear direction for the information flow. 
> Feed-forward 네트워크는 acyclic graph이다. 즉, 정보의 흐름이 명백하다. (앞에서 뒤로)

For any cut separating the inputs from the outputs, one can access the amount of information passing though the cut. One should avoid bottlenecks with extreme compression. 
> bottlenecks으로 인한 정보의 큰 영향을 줌

In general the representation size should gently decrease from the inputs to the outputs before reaching the final representation used for the task at hand. 
> 일반적으로는 `representation`의 크기는 입력에서 출력으로 갈수록 점점 작아 져야 한다. 

Theoretically, information content can not be assessed merely by the dimensionality of the representation as it discards important factors like correlation structure; the dimensionality merely provides a rough estimate of information content.

### 

Higher dimensional representations are easier to process locally within a network. 
> 고차원의 representations은 내부적으로(locally) 진행 하기 쉽다. ??

Increasing the activations per tile in a convolutional network allows for more disentangled features. 
> `activations`의 증가시키면 더 많은 Feature를 추출 할수 있다. 

The resulting networks will train faster.
> 이러한 결과의 네트워크는 학습이 더 빠르다. 

### Spatial aggregation

Spatial aggregation can be done over lower dimensional embeddings without much or any loss in representational power. 
> `Spatial aggregation`은 저차원에서 표현력(representational power) 저하 없이 수행 될수 있다. 

For example, before performing a more spread out (e.g. 3 × 3) convolution, one can reduce the dimension of the input representation before the spatial aggregation without expecting serious adverse effects. 
> 예를 들어, 


We hypothesize that the reason for that is the strong correlation between adjacent unit results in much less loss of information during dimension reduction,if the outputs are used in a spatial aggregation context. 
> 우리의 가설로는 이렇게 되는 이유는 다음과 가다. 만약 아웃풋이 spatial aggregation context에 쓰인다면 이웃한 Unit의 strong correlation 때문에 차원 축소가 일어나도 정보을 적게 잃어 버리게 된다. 

Given that these signals should be easily compressible, the dimension reduction even promotes faster learning..

### Balance the width and depth of the network. 
> 네트워크의 깊이와 넓이의 균형을 유지 하라 

Optimal performance of the network can be reached by balancing the number of filters per stage and the depth of the network. 
> 

Increasing both the width and the depth of the network can contribute to higher quality networks.However, the optimal improvement for a constant amount of computation can be reached if both are increased in parallel. 
The computational budget should therefore be distributed in a balanced way between the depth and width of the network.