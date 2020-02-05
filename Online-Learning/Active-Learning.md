# Active Learning 


> 적절한 자기주도 학습 데이터를 찾는

> https://blogs.nvidia.co.kr/2020/01/29/what-is-active-learning/?fbclid=IwAR1RLqrs8GXeTTlN1P_pdSf2mrJViTNXl1SzbNyICtKPAMRQdf98Um6uwqg


목적 : 데이터셋 구축 
정의 : 다양한 데이터를 자동으로 찾는 머신 러닝을 위한 학습용 데이터 선택 방식
장점 : 사람이 직접 큐레이션(curation)하는데 걸리는 시간 보다 훨씬 짧은 시간 안에 더 좋은 데이터 세트를 구축하는 것입니다.

구성 : 
1. 학습을 완료한 모델을 통해 수집된 정보를 확인한 후, 
2. 그 정보 중에서 인지하기 어려운 프레임을 표시하도록 합니다. 
3. 그리고 그 프레임들을 학습용 데이터에 추가합니다. 
4. 이 과정을 반복하면 좀 더 복잡한 상황에서 모델이 물체를 정확하게 인지하는 능력이 향상됩니다.

