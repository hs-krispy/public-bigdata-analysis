## Algorithms

### Feature extraction

- Autoencoder
- Convolutional Autoencoder
- Variational Autoencoder
- <u>Convolutional Variational Autoencoder</u> ???

### manifold

#### TSNE

#### UMAP (Uniform Manifold Approximation and Projection)

- 데이터의 고차원 그래프 표현을 구성한 다음 가능한 한 구조적으로 유사하도록 저차원 그래프를 최적화
- 각 데이터 포인트 주변의 반경을 확장하고 영역이 교차하게 되는 각 포인트들은 연결되어 있다고 판단
- **k번째 가장 가까운 이웃(n-neighbors)이 가까울 때 밀도가 더 높은 것**으로 멀리 있을 때 밀도가 더 낮은 것으로 추정
  - **반경은 저밀도 영역에서 커지고 고밀도 영역에서 작아짐 (고밀도 영역일수록 반경이 천천히 커짐)**
  - 이때, k는 적절한 로컬 반경을 찾기 위한 밀도 추정에 대한 hyperparameter임
  - **k가 크면 더 많은 전역 구조가 보존**되며 **작으면 반경이 감소하고 로컬 구조가 더 많이 보존**
  - 자동으로 최적을 찾는 방법은 없음
- **데이터 포인트간의 연결에는 가중치(연결 확률)가 적용**되며, 멀리 떨어진 지점에는 가중치가 덜 부여됨 (반경이 커짐에 따라 그래프를 fuzzy(경계를 불분명)하게 만들어서 점 사이의 연결 정도를 줄임)

<img src="https://user-images.githubusercontent.com/58063806/144061637-07e6bef3-54bc-44fc-a531-ff9c6254a2a6.png" width=50% /><img src="https://user-images.githubusercontent.com/58063806/144062006-4506f4e4-f8a9-4469-90a2-270a8edf0514.png" width=50% />

- min_dist : 저차원 공간에서 포인트 간의 최소거리
  - 이 값이 작을수록 포인트들이 촘촘하게 무리지어 있게 됨

<img src="https://user-images.githubusercontent.com/58063806/144062625-c39795ea-9aae-42f9-8890-4768212f2753.png" width=50% /><img src="https://user-images.githubusercontent.com/58063806/144062773-61f695b3-d2be-42e0-82b1-fac27931ab84.png" width=50%/>

- **n-neighbors 값이 작을 때, 가짜 군집이 관찰될 수 있으며** 하이퍼파라미터의 선택이 중요한 만큼 **다양한 하이퍼파라미터로 여러번 실행하는 것이 중요**

- [참고문헌 및 이미지 출처](https://pair-code.github.io/understanding-umap/)

#### ISOMAP

#### LLE

### cluster

#### K-means

#### GMM

**E-M algorithm (기댓값-최대화 알고리즘)**

- 잠재 변수가 있는 상태에서 최대 가능도 추정을 수행하는 접근 방식

- **Expectation step (E - step)** 

  - **모수들이 주어졌을 때, 샘플들이 각 클러스터에 속할 확률을 계산하고 이에 따라 클러스터에 할당**하는 과정

  - > **parameters**
    >
    > - ϕ(phi) : 클러스터 weight (추정확률, **prior**)
    >
    > - μ, Σ : 평균과 분산 (공분산 행렬)
    > - 처음에는 랜덤하게 초기화된 모수들이 주어짐 
    > - sklearn에서는 kmeans를 default로 mean과 weight를 초기화  

  - 샘플 x<sup>(i)</sup>가 주어진 모수들로 구성되는 확률 분포(클러스터, j)에 속할 가능도(Likelihood, PDF 값)와 각 클러스터에 대한 사전확률(prior)을 이용, 특정 클러스터에 속할 때 값을 각 클러스터에 속할 때 값의 합으로 나누면 각 클러스터에 속할 확률이 됨

<img src="https://user-images.githubusercontent.com/58063806/144253152-2810a04c-ec61-428d-82fe-cd22e1bdcb45.png" width=60% />

- **Maximization step (M - step)**

  - E - step에서 할당된 샘플들을 바탕으로 **모수들을 업데이트**

  - ϕ는 각 클러스터 별로 샘플들이 속할 확률의 평균치

  - MLE (Maximum likelihood estimation)

    - 관찰된 데이터를 생성할 가능성을 최대화하는 모수를 추정

    - 데이터를 관찰할 총 확률, 모든 관찰된 데이터의 공통 확률 분포 (각 데이터는 독립적으로 생성된다고 가정)

    - 모든 데이터(x)에 대해 정규분포의 PDF 값을 모두 곱한 식의 결과가 최대가 되는  μ, Σ 값을 추정

    - > 자연로그를 취해서 합의 형태로 변환 후 미분값이 0이 되는 μ, Σ의 log likelihood 추정

    - EX) x : 9, 9.5, 11

<img src="https://user-images.githubusercontent.com/58063806/144256775-c78e6e3f-adc5-44ce-ac2d-c4ae4f335a2c.png" width=70% />

> 다변량 정규분포의 PDF
>
> <img src="https://user-images.githubusercontent.com/58063806/144257886-d91c7354-1647-4fa9-a5a1-5a0e074fcab4.png" width=60% />

- 장점

  - M - step에 대한 해결책은 닫힌 형태로 존재하는 경우가 많음
  - 매 반복마다 우도의  값이 증가하는 것이 항상 보장됨
- 단점

  - 수렴이 느림
  - 로컬 최적값으로만 수렴
    - 전진 확률과 후진 확률을 모두 고려			
  - 클러스터 당 데이터가 충분하지 않으면 공분산 행렬을 추정하기가 어려워지며 알고리즘이 수렴하지 않고 발산하게됨 

[MLE (Maximum likelihood estimation) 참고](https://towardsdatascience.com/probability-concepts-explained-maximum-likelihood-estimation-c7b4342fdbb1)

[EM & Gaussian mixture 참고-1](https://angeloyeo.github.io/2021/02/08/GMM_and_EM.html)

[EM & Gaussian mixture 참고-2](https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95)

[Gaussian mixture 참고](https://scikit-learn.org/stable/modules/mixture.html)

- Spectral Clustering

### K-L divergence (쿨백-라이블러 발산)

- 두 확률 분포를 비교
- 확률분포 P가 있을 때, 샘플링 과정에서 그 분포를 근사적으로 표현하는 확률분포 Q를 P 대신 사용할 경우 엔트로피 변화를 의미 
- 이산확률변수와 연속확률변수의 경우

<img src="https://user-images.githubusercontent.com/58063806/144260970-13116f15-e32b-4515-a91e-f1c911872bf3.png" width=25% />

- 원래 분포가 가지는 엔트로피 H(P)와 P 대신 Q를 사용할 때의 교차 엔트로피(cross entropy) H(P, Q)의 차이

<img src="https://user-images.githubusercontent.com/58063806/144261395-0f2f46ca-dc4f-4ef2-9dbf-042298efe7c4.png" width=30% />

[출처](https://ko.wikipedia.org/wiki/%EC%BF%A8%EB%B0%B1-%EB%9D%BC%EC%9D%B4%EB%B8%94%EB%9F%AC_%EB%B0%9C%EC%82%B0)
