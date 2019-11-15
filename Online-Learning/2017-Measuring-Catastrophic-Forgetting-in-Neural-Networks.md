# [Measuring Catastrophic Forgetting in Neural Networks](https://arxiv.org/pdf/1708.02072.pdf)


Abstract - 딥러닝은 인지 시스템에 좋은 성능을 보이지만 한번 학습이 되면 새로운 분야를 학습 하는건 어렵다 ` Deep neural networks are used in many state-of-the-art systems for machine perception. Once a network is trained to do a specific task, e.g., bird classification, it cannot easily be trained to do new tasks,`
-  e.g., incrementally learning to recognize additional bird species or learning an entirely different task such as flower recognition. 

새 학습이 더해지먼 **catastrophically forgetting**라는 이전 학습을 잃어 버리게 된다. `When new tasks are added, typical deep neural networks are prone to catastrophically forgetting previous tasks. Networks that are capable of assimilating new information incrementally, much like how humans form new memories over time, will be more efficient than retraining the model from scratch each time a new task needs to be learned. `

문제 해결을 위한 연구가 있었다. 하지만 비교는 없으며, 작은 문제(MNIST)에만 적용이 되어 왔다. `There have been multiple attempts to develop schemes that mitigate catastrophic forgetting, but these methods have not been directly compared, the tests used to evaluate them vary considerably, and these methods have only been evaluated on small-scale problems (e.g., MNIST). `

본 논문에서는 성능평가 메트릭을 소개 하고 5가지로 분류된 기존 연구에 대하여 테스트 하였다. `In this paper, we introduce new metrics and benchmarks for directly comparing five different mechanisms designed to mitigate catastrophic forgetting in neural networks: `
- regularization, 
- ensembling, 
- rehearsal, 
- dual-memory, 
- and sparse-coding. 

Our experiments on real-world images and sounds show that the mechanism(s) that are critical for optimal performance vary based on the incremental training paradigm and type of data being used, but they all demonstrate that the catastrophic forgetting problem has yet to be solved

## 1. Introduction

딥러닝은 많은 분야에서 좋은 결과를 보이고 있다. `While the basic architecture and training algorithms behind deep neural networks (DNNs) are over 30 years old, interest in them has never been greater in both industry and the artificial intelligence research community. Owing to far larger datasets, increases in computational power, and innovations in activation functions, DNNs have achieved near-human or super-human abilities on a number of problems, including image classification (He et al. 2016), speech-to-text (Khilari and Bhope 2015), and face identification (Schroff, Kalenichenko, and Philbin 2015). These algorithms power most of the recent advances in semantic segmentation (Long, Shelhamer, and Darrell 2015), visual question answering (Kafle and Kanan 2017), and reinforcement learning (Mnih et al. 2013). `

하지만 새로운 학습을 진행시 발생 하는  파괴적 망각(catastrophically forgetting)은 해결 하지 못하고 있다.`While these systems have become more capable, the standard multi-layer perceptron (MLP) architecture and typical training algorithms cannot handle incrementally learning new tasks or categories with-out catastrophically forgetting previously learned training data. Fixing this problem is critical to making agents that incrementally improve after deployment. `

일부 특수한 경우에서는 전체 네트워크를 재 학습 하는 방법으로 망각문제 해결을 할수도 있다. 하지만 재 학습은 느리다. `For non-embedded or personalized systems, catastrophic forgetting is often overcome simply by storing new training examples and then re-training either the entire network from scratch or possibly only the last few layers. In both cases, retraining uses both the previously learned examples and the new examples, randomly shuffling them so that they are independent and identically distributed (iid). Retraining can be slow, especially if a dataset has millions or billions of in instances. `

망각문제는 30년 전에 소개 되었으며 많은 연구가 진행 되었다. `Catastrophic forgetting was first recognized in MLPs almost 30 years ago (McCloskey and Cohen 1989). Since then, there have been multiple attempts to mitigate this phenomenon (Hinton and Plaut 1987; Robins 1995; Goodrich and Arel 2014; Draelos et al. 2016; Ren et al. 2017; Fernando et al. 2017; Kirkpatrick et al. 2017). However, these methods vary considerably in how they train and evaluate their models and they focus on small dataset, e.g., MNIST. `

대용량의 데이터에 대한 테스트를 진행 하였따. `It is not clear if these methods will scale to larger datasets containing hundreds of categories. In this paper, we remedy this problem by providing a comprehensive empirical review of methods to mitigate catastrophic forgetting across a variety of new metrics. `

망각 현상이 비지도 프레임워크에서 발생.....While catastrophic forgetting occurs in unsupervised frameworks (Draelos et al. 2016; Goodrich and Arel 2014; Triki et al. 2017), we focus on supervised classification. 

```
Draelos, T. J.; Miner, N. E.; Lamb, C. C.; Vineyard, C. M.; Carlson, K. D.; James, C. D.; and Aimone, J. B. 2016. Neurogenesis deep learning. arXiv:1612.03770.
Goodrich, B., and Arel, I. 2014. Unsupervised neuron selection for mitigating catastrophic forgetting in neural networks. In IEEE 57th Int. Midwest Symposium on Circuits and Systems (MWSCAS), 2014, 997–1000. IEEE.
Triki, A. R.; Aljundi, R.; Blaschko, M. B.; and Tuytelaars, T. 2017. Encoder based lifelong learning. arXiv:1704.01920
```

본 논문의 기여는 다음과 같다. `Our major contributions are:`
- 여러 방식이 있지만 망각 현상은 해결 되지 않았음을 보였다. `We demonstrate that despite popular claims (Kirkpatrick et al. 2017), catastrophic forgetting is not solved. `
- 새 테스트 방법론을 사용하고, MNIST등에서 적용되는 방법도 대용량에서는 적용되지 않음을 보였다. `We establish new benchmarks with novel metrics for measuring catastrophic forgetting. `
	- Previous work has focused on MNIST, which contains low-resolution images and only 10 classes. Instead, 
	- we use real-world image/audio classification datasets containing 100-200 classes. 
	- We show that, although existing models perform well on MNIST for a variety of different incremental learning problems, performance drops significantly with more challenging datasets.
- 기술을 5가지로 분류 하였다. `We identified five common mechanisms for mitigating catastrophic forgetting: `
	- 1) regularization, 
	- 2) ensembling, 
	- 3) rehearsal, 
	- 4) dual-memory models, and 
	- 5) sparsecoding. 
	- Unlike previous work, we directly compare these distinct approaches.

### Problem Formulation

본 논문에서는 **분류**작업을 하는 딥러닝 네트워에서 **순차학습**을 진행시 발생하는 **망각**현상에 대하여 살펴 보았다. `In this paper, we study catastrophic forgetting in MLP-based neural networks that are incrementally trained for classification tasks. `

In our setup, 
- the labeled training dataset D is organized into T study sessions (batches), i.e., D = {B_t}^T_{t=1} . 
- Each study session B_t consists of N_t labeled training data points, 
	- i.e., B_t =  (xj ,yj) Nt j=1 , where xj ∈ R d and yj is a discrete label. 
- N_t is variable across sessions. 

The model is only permitted to learn sessions sequentially, in order. 

At time t the network can only learn from study session B_t ; 
	- however, models may use auxiliary memory to store previously observed sessions, 
	- but this memory use must be reported. 
- We do not assume sessions are iid, 
	- 즉, 학습데이터에 한 카테고리 정보만 있을수도 있다. `e.g., some sessions may contain data from only a single category. `

In between sessions, the model may be evaluated on test data. Because this paper’s focus is catastrophic forgetting, we focus less on representation learning and obtain feature vectors using embeddings from pre-trained networks. 

Note that in some other papers, new sessions are called new ‘tasks.’ 

We refer to the first study session as the model’s ‘base set knowledge.’


## Why Does Catastrophic Forgetting Occur?

망각 현상이 발생 하는 원인은 **stability-plasticity dilemma** 때문이다. `Catastrophic forgetting in neural networks occurs because of the stability-plasticity dilemma (Abraham and Robins 2005). `

설명 : The model requires sufficient plasticity to acquire new tasks, but large weight changes will cause forgetting by disrupting previously learned representations. Keeping the network’s weights stable prevents previously learned tasks from being forgotten, but too much stability prevents the model from learning new tasks. 

기존 2가지 접근법. `Prior research has tried to solve this problem using two broad approaches. `
- 기존 학습 파라미터와 새 학습 파라 미터 나누기 `The first is to try to keep new and old representations separate, which can be done using distributed models, regularization, and ensembling. `
- 기존 Task도 같이 학습 `The second is to prevent the forgetting of prior knowledge simply by training on the old tasks (or some facsimile of them) as well as new tasks, thereby preventing the old tasks from being forgotten. `
	- Besides requiring costly relearning of previous examples and additional storage, this scheme is still not as effective as simply combining the new data with the old data and completely re-training the model from scratch. 
	- This solution is inefficient as it prevents the development of deployable systems that are capable of learning new tasks over the course of their lifetime.

## Previous Surveys

French (1999) exhaustively reviewed mechanisms for preventing catastrophic forgetting that were explored in the 1980s and 1990s. 


Goodfellow et al. (2013) compared different activation functions and learning algorithms to see how they affected catastrophic forgetting, but these methods were not explicitly designed to mitigate catastrophic forgetting. 
- The authors concluded that the learning algorithms have a larger impact, which is what we focus on in our paper. 
- They sequentially trained a network on two separate tasks using three different scenarios: 
	- 1) identical tasks with different forms of input, 
	- 2) similar tasks, and 
	- 3) dissimilar tasks. 
- We adopt a similar paradigm, but our experiments involve a much larger number of tasks. We also focus on methods explicitly designed to mitigate catastrophic forgetting. 


Soltoggio, Stanley, and Risi (2017) reviewed neural networks that can adapt their plasticity over time, which they called Evolved Plastic Artificial Neural Networks. 
- Their review covered a wide-range of brain-inspired algorithms and also identified that the field lacks appropriate benchmarks. 
- However, they did not conduct any experiments or establish benchmarks for measuring catastrophic forgetting. 
- We remedy this gap in the literature by establishing large-scale benchmarks for evaluating catastrophic forgetting in neural networks, and we compare methods that use five distinct mechanisms for mitigating it.

## Mitigating Catastrophic Forgetting

5가지 분류로 나누어 다음장에 설명 하였다. `While not exhaustive, we have identified five main approaches that have been pursued for mitigating catastrophic forgetting in MLP-like architectures, which we describe in the next subsections. `

> 모델에 대한 상세한 설명은 [Experimental Setup] 섹션에 기술 하였다. `We describe the models we have selected in greater detail in the Experimental Setup section.`

### A. Regularization Methods (eg. EWC:2017)

Regularization methods **add constraints** to the network’s weight updates, so that a new session is learned without interfering with prior memories. 

Hinton and Plaut (1987) implemented a network that had both ‘fast’ and ‘slow’ training weights. 
- The fast weights had high plasticity and were easily affected by changes to the network, 
- and the ‘slow’ weights had high stability and were harder to adapt. 
- This kind of dual-weight architecture is similar in idea to dual-network models, but has not been proven to be sufficiently powerful to learn a large number of new tasks. 

Elastic weight consolidation (EWC) (Kirkpatrick et al. 2017) adds a constraint to the loss function that directs plasticity away from weights that contribute the most to previous tasks. 

> We use EWC to evaluate the regularization mechanism.

### B. Ensemble Methods (eg. PathNET:2017)

Ensemble methods attempt to mitigate catastrophic forgetting either by explicitly or implicitly **training multiple classifiers** together and then combining them to generate the final prediction. 

#### 방법 #1 : Explicitly

학습마다 새로운 서브 네트워크 생성 하여 진행 : For the **explicit methods**, such as Learn++ and TradaBoost, this prevents forgetting because an entirely **new sub-network** is trained for a new session (Polikar et al. 2001; Dai et al. 2007). 
- 메모리 사용량 증가 : However, **memory usage** will scale with the number of sessions, which is highly non-desirable. 
- 네트워크 재 사용성 없음 Moreover, this prevents portions of the network from being reused for the new session. 

메모리 사용량을 줄기이 위한 노력도 있음 : Two methods that try to alleviate the memory usage problem are **Accuracy Weighted Ensembles** and **Life-long Machine Learning** (Wang et al. 2003; Ren et al. 2017). 
- These methods automatically decide whether a sub-network should be removed or added to the ensemble.

#### 방법 #2 : Implicit

**PathNet** can be considered as an implicit ensemble method (Fernando et al. 2017). 
- It uses a genetic algorithm to find an optimal path through a fixed-size neural network for each study session. 
- The weights in this path are then frozen; so that when new sessions are learned, the knowledge is not lost. 

In contrast to the explicit ensembles, the base network’s size is fixed and it is possible for learned representations to be re-used which allows for smaller, more deployable models. 

The authors showed that PathNet learned subsequent tasks more quickly, but not how well earlier tasks were retained. 

> We have selected PathNet to evaluate the ensembling mechanism, and we show how well it retains pre-trained information.


### C. Rehearsal Methods (eg. GeppNet:2016)

Rehearsal methods try to mitigate catastrophic forgetting by **mixing data** from earlier sessions with the current session being learned (Robins 1995). 

과거 데이터 저장 하는 부하가 발생 한다. `The cost is that this requires storing past data, which is not resource efficient. `

Pseudo-rehearsal methods use the network to generate pseudopatterns (Robins 1995) that are combined with the session currently being learned. 
- Pseudopatterns allow the network to stabilize older memories without the requirement for storing all previously observed training data points. 

Draelos et al. (2016) used this approach to incrementally train an autoencoder, where each session contained images from a specific category. 
- After the autoencoder learned a particular session, they passed the session’s data through the encoder and stored the output statistics. 
- During replay, they used these statistics and the decoder network to generate the appropriate pseudopatterns for each class.

The **GeppNet** model proposed by Gepperth and Karaoguz (2016) reserves its training data to replay after each new class was trained. 
- This model used a self-organizing map (SOM) as a hidden-layer to topologically reorganize the data from the input layer (i.e., clustering the input onto a 2-D lattice). 

> We use this model to explore the value of rehearsal.

### D. Dual-Memory Models

Dual-memory models are inspired by memory consolidation in the mammalian brain, which is thought to **store memories in two distinct neural networks**. 

Newly formed memories are stored in a brain region known as the **hippocampus**. These memories are then slowly transferred/consolidated to the pre-frontal cortex during sleep. 

Several algorithms based on these ideas have been created. Early work used fast (hippocampal) and slow (cortical) training networks to separate pattern-processing areas, and they passed pseudopatterns back and forth to consolidate recent and remote memories (French 1997). 

일반적으로 이 방식은 리허설 방식과 같이 사용된다. 그렇다고 모든 리허설 방식이 듀얼 메모리 방식인건 아니다. `In general, dual-memory models incorporate rehearsal, but not all rehearsal-based models are dualmemory models.`

Another model proposed by Gepperth and Karaoguz (2016), which we denote GeppNet+STM, stores new inputs that yield a highly uncertain prediction into a short-term memory (STM) buffer. 
- This model then seeks to consolidate the new memories into the entire network during a separate sleep phase. 
- They showed that GeppNet+STM could incrementally learn MNIST classes without forgetting previously trained ones. 

> We use GeppNet and GeppNet+STM to evaluate the dual-memory approach.

### E. Sparse-Coding Methods

망각현상은 새 학습값이 기존 학습을 **간섭**할때 발생 한다. `Catastrophic forgetting occurs when new internal representations interfere with previously learned ones (French 1999). `

이 방식에서는 이러한 **간섭**을 줄이는 방법을 제안 하였다. `Sparse representations can reduce the chance of this interference; however, sparsity can impair generalization and ability to learn new tasks (Sharkey and Sharkey 1995).`

Two models that implicitly use sparsity are CALM and ALCOVE. 
- To learn new data, CALM searches among competing nodes to see which nodes have not been committed to another representation (Murre 2014). 
- ALCOVE is a shallow neural network that uses a sparse distance-based representation, which allows the weights assigned to older tasks to be largely unchanged when the network is presented with new data (Kruschke 1992). 

The Sparse Distributed Memory (SDM) is a convolution-correlation model that uses sparsity to reduce the overlap between internal representations (Kanerva 1988). 

CHARM and TODAM are also convolution-correlation models that use internal codings to ensure that new input representations remain orthogonal to one another (Murdock 1983; Eich 1982).

The Fixed Expansion Layer (FEL) model creates sparse representations by fixing the network’s weights and specifying neuron triggering conditions (Coop, Mishtal, and Arel 2013). 
- FEL uses excitatory and inhibitory fixed weights to sparsify the input, which gates the weight updates throughout the network. 
- This enables the network to retain prior learned mappings and reduce representational overlap. 

> We use FEL to evaluate the sparsity mechanism.

## Experimental Setup

## Experiments and Results

![](https://i.imgur.com/A2ex16X.png)

## Discussion

테스트 결과 위 5가지 모두 망각 문제를 해결 하진 못했다. 단지 상대적으로 성능이 좋은 알고리즘이 있을 뿐이다. `In our paper we introduced new metrics and benchmarks for measuring catastrophic forgetting. Our results reveal that none of the methods we tested solve catastrophic forgetting,  while also enabling the learning of new information. Table 3 summarizes these results for each of our experiments by averaging Ω_{al}l over all datasets. While no method excels at incremental learning, some perform better than others. `

![](https://i.imgur.com/M5TgamF.png)

PathNet performed best overall on the data permutation experiments, with the exception of CUB-200. However, PathNet requires being told which session each test instance is from, whereas the other models do not use this information. This may give it an unfair advantage. PathNet works by locking the optimal path for a given session. Because permuting the data does not reduce feature overlap, the model requires more trainable weights (less feature sharing) to build a discriminative model, causing PathNet to saturate (freeze all weights) more quickly. When PathNet reaches the saturation point, the only trainable parameters are in the output layer. While EWC was the second best performing method in the permutation experiments, it only redirects plasticity instead of freezing trainable weights. 

Both GeppNet variants performed best at incremental class learning. These models make slow, gradual changes to the network that are inspired by memory consolidation during sleep. For these models, the SOM-layer was fixed to 23×23 to have the same number of trainable parameters as the other models. With 100-200 classes, this corresponds to 2-5 hidden layer neurons per class respectively. The experiments on MNIST in Gepperth and Karaoguz (2016) used 90 hidden-layer neurons per class, so their performance may improve if their model capacity was significantly increased, but this would demand more memory and computation. 

EWC performed best on the multi-modal experiment. This may be because features between the two modalities are non-redundant. We hypothesize that EWC is a better choice for separating non-redundant data and PathNet may work well when working with data that has different, but not entirely dissimilar, representations. To explore this, we used the Fast Correlation Based Filter proposed by Yu and Liu (2003) to show the features in MNIST and AudioSet are more redundant than those in CUB-200 (see supplemental material). The performance of EWC and PathNet for both the data permutation and multi-modal experiments are consistent with this hypothesis. 

> 중략 

## Conclusion

In this paper, we developed new metrics for evaluating catastrophic forgetting. We identified five families of mechanisms for mitigating catastrophic forgetting in DNNs. We found that performance on MNIST was significantly better than on the larger datasets we used. 

Using our new metrics, experimental results (summarized in Table 5) show that 
- 1) a combination of rehearsal/pseudo-rehearsal and dual-memory systems are optimal for learning new classes incrementally, and 
- 2) regularization and ensembling are best at separating multiple dissimilar sessions in a common DNN framework. 

Although the rehearsal system performed reasonably well, it required retaining all training data for replay. 
- This type of system may not be scalable for a real-world lifelong learning system; 
- however, it does indicate that models that use pseudorehearsal could be a viable option for realtime incremental learning systems. 

Future work on lifelong learning frameworks should involve combinations of these mechanisms. While some models perform better than others in different scenarios, our work shows that catastrophic forgetting is not solved by any single method. This is because there is no model that is capable of assimilating new information while simultaneously and efficiently preserving the old. We urge the community to use larger datasets in future work.
