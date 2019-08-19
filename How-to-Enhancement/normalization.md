# 데이터 전처리 (데이터 정규화 & 표준화)

- 학습 절차 전에 수행 
- 데이터의 어떤 `경향`을 제거하여 동등한 환경의 데이터를 만드는 작업 

## 1. 정규화(Normalization)
![](/assets/nor.png)
* 전체 구간을 0~100으로 설정하여 데이터를 관찰 
* 데이터 군 내에서 특정 데이터가 가지는 위치 관찰
* 결과는 comparable range안에 들어 오게 됨(0~1 OR -1 ~ +1 )


정규화는 데이터의 진폭을 제거 하여 분포의 모양에 초점을 맞추는 것 Normalization is performed on data to remove amplitude variation and only focus on the underlying distribution shape.


### 정규화 방법 

> 출처 : [A.I. SERIES PART 1 - NORMALIZING DATA]](https://www.skcript.com/svr/normalizing-data-artificial-intelligence/)


1. Min Max Normalization
2. Max Normalization
3. L1 Normalization (Least Absolute Deviation or LAD)
4. L2 Normalization (Least Square Error or LSE)
5. Z-Score




## 2. 표준화
![](/assets/stamd.png)
* 평균을 기준으로 얼마나 떨어져 있는지 관찰
* 2개 이상의 데이터 단위가 다를때 대상 데이터를 같은 기준으로 볼수 있게 함(나이 vs 키)

