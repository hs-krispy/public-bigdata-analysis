



## Algorithms

### Feature extraction

- Autoencoder

- Variational Autoencoder

- Convolutional Autoencoder

- <u>Convolutional Variational Autoencoder</u> ???

#### K-L divergence (쿨백-라이블러 발산)

- 두 확률 분포를 비교
- 확률분포 P가 있을 때, 샘플링 과정에서 그 분포를 근사적으로 표현하는 확률분포 Q를 P 대신 사용할 경우 엔트로피 변화를 의미 

$$
D_{KL}(P||Q) = \sum_{i}P(i)log\frac{P(i)}{Q(i)}\space \text 이산확률변수
$$

$$
D_{KL}(P||Q) = \int_{-∞}^∞ p(x)log\frac{p(x)}{q(x)} \space \text 연속확률변수
$$

- 원래 분포가 가지는 엔트로피 H(P)와 P 대신 Q를 사용할 때의 교차 엔트로피(cross entropy) H(P, Q)의 차이
  $$
  D_{KL}(P||Q) = H(P, Q) - H(P)
  $$

  $$
  D_{KL}(P||Q) = -\sum_{x}p(x)log\space q(x) + \sum_{x}p(x)log\space p(x)
  $$

#### E-M algorithm (기댓값-최대화 알고리즘)

- 잠재 변수 모형은 관측되지 않은 변수(잠재 변수)와 관련된 일반적인 확률 분포 형식을 알고 있는 경우, 데이터 집합에서 이러한 결측값을 예측하는 데에 사용
- 잠재 변수의 관측 가능한 표본을 사용하여 학습에 대해 관측할 수 없는 표본의 값을 예측
- Expectation step (E - step) 

  - 데이터 세트의 관찰된 데이터를 사용하여 누락된 데이터의 값을 추정하거나 추측
  - 이 단계 후에는 결측값이 없는 완전한 데이터를 얻을 수 있음
- Maximization step (M - step)

  - E - step에서 준비된 완전한 데이터를 사용하여 파라미터를 업데이트
- 장점

  - M - step에 대한 해결책은 닫힌 형태로 존재하는 경우가 많음
  - 매 반복마다 우도의  값이 증가하는 것이 항상 보장됨
- 단점

  - 수렴이 느림
  - 로컬 최적값으로만 수렴
  - 전진 확률과 후진 확률을 모두 고려

#### manifold

- TSNE
- UMAP
- ISOMAP
- LLE

#### cluster

- K-means
- GMM
- Spectral Clustering