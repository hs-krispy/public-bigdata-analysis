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

### K-L divergence (쿨백-라이블러 발산)

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

### E-M algorithm (기댓값-최대화 알고리즘)

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

### cluster

- K-means
- GMM
- Spectral Clustering

