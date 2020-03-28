# [Anomaly Detection in SUALAB] http://research.sualab.com/introduction/review/2020/01/30/anomaly-detection-overview-1.html?fbclid=IwAR3mxr6V65F9yTthrSLxuy-n_CrMZxtXhL9fHdlTbVusWQlN1cdlNe1ynTU)
- Anormaly Detection이란 Normal, Abnormal 샘플을 구별해내는 문제를 의미함
## 학습 방법에 따른 분류
- Supervised Anormaly Detection
    - 정상/비정상 샘플이 모두 주어진 경우
    - 비정상 샘플을 다양하게 보유할수록 더 높은 성능을 달성할 수 있다.
    - Class-Imbalance 문제를 자주 겪에 된다.
    - Data augmentation, Loss function, batch sampling 등 다양한 연구가 수행되고 있다.
        - 장점: 양/불 판정 정확도가 높다
        - 단점: 비정상 샘플을 취득하는데 시간, 비용이 많이 들고 class-imbalance 문제를 해결해야 한다.
- Semi-supervised (one-class) Anomaly Detection
    - One-Class Classification(혹은 Semi-supervised learning) Class-Inbalance가 매우 심한 경우 정상 sample만 이용해서 학습함.
    - 핵심 아이디어: 정상 샘플들을 둘러싸는 discriminative boundary를 설정하고, 이 boundary를 최대합 좁혀 boundary 밖에 있는 sample들을 모두 비정상으로 간주함.
    - One-class SVM이 대표적인 방법론으로 알려져 있음
    - [Deep SVDD](http://data.bit.uni-bonn.de/publications/ICML2018.pdf) 논문이 딥러닝 기반의 방법론이다 
        - 장점: 비교적 활발한 연구, 정상 sample만 있어도 학습 가능
        - 단점: Supervised Anormaly Detection 방법론과 비교했을 때 상대적으로 양/불 판정 정확도가 낮다.

- Unsupervised Anomaly Detection
    - 레이블 없이 PCA나 Autoencoder 기반의 방법론으로 사용
    - Autoencoder를 이용하면 데이터에 대한 labeling 없이 주성분이 되는 정상 영역의 특징을 학습할 수 있다.
    - 정상 데이터를 넣으면 input과 output 차이가 거의 발생하지 않는 반면, 비정상 sample을 넣으면 정상 sample처럼 복원하기 때문에 Input과 output의 차이를 구하는 과정에서 차이가 도드라지게 발생하므로 비정상 sample을 검출할 수 있다.
    - 단점: 압축정도와 같은 하이퍼파라미터에대해 복원 성능이 좌우되기 때문에 양/불 판정 정확도가 Supervised Anormaly Detection에 비해 다소 불안정하다는 단점이 존재. (loss function/difference map을 어떻게 계산할 지등등에 따라서 성능이 크게 달라 질 수 있다.)
    -장점: 별도의 labeling 과정 없이 어느정도 성능을 낼 수 있다.

2. 비정상 sample 정의에 따른 분류
- Novelty Detection / Outlier Detection
- 강아지를 normal class라고 할때, 한번도 본적 없는 새로운 강아지가 등장하는 경우, 이러한 sample을 Novel sample, Unseen sample등으로 부를 수 있다.
- 강아지와 전혀 상관없는 데이터를 Outlier Detection이라고 볼 수 있다.
- 경계는 명확하지 않으니 잘구분하면서 보기

3. 정상 sample의 class 개수에 따른 분류
- 정상 sample이 multi-class인 경우
- 보통 이러한 경우 정상 sample이라는 표현대신 In-distribution sample이라는 표현을 사용한다.

## 다양한 사례
- [survay anormal detection](https://arxiv.org/abs/1901.03407)
- Cyber-Intrusion Detection: 컴퓨터 시스템 상에 침입을 탐지하는 사례. 주로 시계열 데이터를 다루며 RAM, file system, log file 등 일련의 시계열 데이터에 대해 이상치를 검출하여 침입을 탐지함.
- Fraud Detection: 보험, 신용, 금융 관련 데이터에서 불법 행위를 검출하는 사례. 주로 tabular 데이터를 다룬다.
- Malwave Detection: Malware(악성코드)를 검출해내는 사례. Classification과 clustering이 주로 사용되며 Malware tabular 데이터를 그대로 이용하기도 하고 이를 gray scale image로 변환하여 이용하기도 함.
- Medical Anomaly Detection: 의료 영상. 뇌파 기록 등의 의학 데이터에 대한 이상치 탐지 사례. 주로 신호 데이터와 이미지 데이터를 다룸으로, 난이가도 높음 
- Social Networks Anormaly Detection: Social Network 상의 이상치들을 검출하는 사례. 주로 Text 데이터를 다루며 Text를 통해 스팸 메일, 비매너 이용자, 허위 정보 유포자 등을 검출함.
- Log Anomaly Detection: 시스템이 기록한 Log를 보고 실패 원인을 추적하는 사례. 주로 Text 데이터를 다루며 pattern matching 기반의 단순한 방법론을 사용하여 해결할 수 있지만 failure message가 새로운 것이 계속 추가, 제외가 되는 경우에 딥러닝 기반 방법론을 사용하는 것이 효과적임
- IoT Big-Data Anomaly Detection: 사물 인터넷에 주로 사용되는 장치, 센서들로부터 생성된 데이ㅓ에 대해 이상치를 탐지하는 사례. 주로 시계열 데이터를 다루며 여러 장치들이 복합적으로 구성이 되어있기 때문에 난이도가 높음
- Industrial Anomaly Detection: 산업 속 제조업 데이터에 대한 이상치를 탐지하는 사례. 각종 제조업 도메인 이미지 데이터에 대한 외관 검사, 장비로부터 측정된 시계열 데이터를 기반으로 한 고장 예측 등 다양한 적용 사례가 있으며, 외관상에 발생하는 결함과, 장비의 고장 등의 비정상적인 sample이 굉장히 적은 수로 발생하지만 정확하게 예측하지 못하면 큰 손실이 유발하기 때문에 난이도가 높음
- Video Surveillance: 비디오 영상에서 이상한 행동이 발생하는 것을 모니터링함.

---
#### reference
[SUA tech blog](http://research.sualab.com/introduction/review/2020/01/30/anomaly-detection-overview-1.html?fbclid=IwAR3mxr6V65F9yTthrSLxuy-n_CrMZxtXhL9fHdlTbVusWQlN1cdlNe1ynTU)