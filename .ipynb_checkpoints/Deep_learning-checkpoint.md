Deep learning Methods
---
### INDEX
- 1. CNN
- 2. RNN

---
#### 0. 딥러닝 학습관련
- 정확도를 높이려면?
    - ensemble
    - learning rate decay
    - data augumentation
- 깊게 학습하는 이유
    - 신경망의 매개변수 수가 줄어든다
    - 작은 필터를 겹쳐 신경망을 깊게 할때 넓은 receptive field를 소화할 수 있다. 층을 거듭하면서 표현력이 개선되며 이는 활성화 함수가 신경망에서 '비선형'힘을 가하고, 비선형 함수가 겹치면서 더 복잡한 것도 표현할 수 있다.
- 딥러닝 고속화
    - GPU 
        - GPU: 큰 행렬의 곱 / 대량 병렬 연산이 특기
        - CPU: 연속적인 복잡한 계산을 잘 처리함
        - 합성곱 연산 (matrix calculation) 파트가 시간이 오래 걸림. 이쪽을 빨리 처리하는게 중요 (GPU가 유리)
    - 분산학습
        - 수평 확장(scale out)하는 분산 학습이 중요
        - 계산을 어떻게 분산시키냐가 어려운 문제다.
            - 컴퓨터 사이의 통신과 데이터 동기화 등등 -> TF/Pytorch!
    - 연산 정밀도와 비트 줄이기
        - 메모리 용량과 버스 대역폭등이 병목이 될 수 있다.
            - 메모리용량: 대향의 가중치 매개변수와 중간 데이터를 메모리에 저장해야함.
            - 버스 대역폭 측면: GPU(or CPU)의 버스를 흐르는데 데이터가 많아져 한계를 넘어서면 병목이 된다.
                - 그러므로, 네트워크로 주고 받는 데이터의 비트수는 최소로 만드는 것이 바람직.
                - * 버스(영어: bus, 문화어: 모선)는 컴퓨터 안의 부품들 간에, 또는 컴퓨터 간에 데이터를 전송하는 통신 시스템이다.
            - 컴퓨터는 주로 64비트나 32비트 부동소수점 수를 사용해 실수를 표현.
                - 많은 비트를 사용할수록 계산 오차 감소
                - 계산에 드는 비용과 메모리 사용량이 늘고 버스 대역폭에 부담
                - 딥러닝은 높은 수치 정밀도를 요구하지 않음
                    - 신경망의 견고성에 따른 특성
                    - 16 bit half-precision만으로도 학습에 문제 없음.
                    - 딥러닝 고속화하기 위해 비트를 줄이는 기술은 중요하며 임베디드용으로 이용할 때 중요한 주제임                    
        
    
---
#### 1. Convolutional Neural Network (CNN)
- C) convolution 연산은 이미지뿐아니라 graph에도 사용된다. 하나의 접근 방식이라고 생각
---
- 기존 ANN : [Affine-ReLU] - [Affine-ReLU] - ... - [Affine-softmax]
    - Fully Connection의 문제점
        - 이미지와 같은 데이터 형상을 무시 : 이미지 픽셀은 바로 옆의 픽셀과(공간적) 연관 이 크다. RGB(색상) 등등
        - 형상을 유지할 수 있음.
        - CNN 도입
    
- CNN : [Conv-ReLU-Pooling]-[Conv-ReLU-Pooling]-...-[Affine-softmax]
    - Concept
        - feature map: convolutional layer의 input/output
        - filter 계산(kernel)이라고도 한다.
        - filter의 window를 일정 간격으로 이동해가며 입력 데이터에 적용
        - bias 도 있는데, 필터로 연산 이후 bias(1x1)을 더해준다(같은 값).
        - Padding
            - 합성곱 연산을 수행하기 전에 입력 데이터 주변을 특정 값(0)으로 채운다. 1짜리면 사방에 한칸씩 0을 채운다.
            - 패딩은 주로 출력 크기를 조절할 목적으로 사용된다.
            - convolution을 계속하면 필터의 크기가 적어진다. 이것들을 패딩으로 채워 입력 데이터의 공간의 크기를 정할 수 있다.
        - Stride
            - 몇칸씩 이동할 것인지
            - 입력 크기 (H,W) / 필터 크기 (FH, FW) / 출력 크기 (OH, OW) / 패딩 (P) / 스트라이트 (S)
                - $OH = \frac{H+2P-FH}{S}+1$
                - $OW = \frac{W+2P-FW}{S}+1$
                - * 정수로 안떨어지면 정수화 시키는 예외처리를 해야함. 보통 딥러닝 프레임워크에는 에러 내지 않도록 구현함.
        - Pooling
            - 세로 가로 방향의 공간을 줄이는 연산
---
- Convolutional 연산
    - 3차원인 경우
        - 입력 데이터의 채널 수 = 필터의 채널 수(depth). 계산하여 연산을 수행 후 하나의 출력값(feature map)을 얻는다.
            - ex) (C, H, W) * (C, FH, FH) -> (1, OH, OW) where C: 채널수
    - 다수의 feature map이 필요한 경우
        - filter를 다수개를 사용함
            - ex) (C, H, w) * (FN, C, FH, FW) -> (FN, OH, OW) where FN: 필터 수
            - bias의 경우는 (FN, 1, 1)을 사용한다.
    - 배치 처리
        - 데이터의 차원을 하나 더 늘려 4차원으로 저장 (데이터수, 패널 수, 높이, 너비)
            - ex) (N, C, H, W) * (FN, C, FH, FW) -> (N, FN, OH, OW) + (FN,1,1) -> (N, FN, OH, OW)
    - Pooling Layer
        - 연산
            - 세로 가로 방향의 공간을 줄이는 연산 (2x2)를 원소 하나로 공간 크기를 줄인다.
                - ex) 2 x 2 max pooling stride 2 : 2x2 영역에서의 최대값을 반환하는 작업을 window size를 2만큼 실시
                - max pooling, average pooling등등이 있다. 보통은 max pooling을 사용한다.
        - 특징
            - 학습해야 할 매개변수가 없다. (average나 max 등등)
            - 채널 수가 변하지 않는다. (feature map의 가로 세로만 달라짐)
            - 입력의 변화에 영향을 적게 받는다 (Robust)
     - 합성곱/풀링 계층 구하기 (어떤 식으로 computation을 실시하는 지)
         - im2col(image to column)을 사용하면 계산을 위해 3차원 to 2차원으롭 변환해서 빠르게 계산
             - 한영역을 한 줄로 만든다.
                 - stride를 적게 잡으면 데이터가 겹치는 부분이 생기므로 데이터가 row가 길어진다
                     - 이는 메모리를 더 많이 소비하는 단점이 있다
                     - 하지만, 연산 속도에서는 탁월함 
             - 필터도 한칼럼이 하나의 필터를 의미하도록 2차원으로 펼치다.
             - 이후 연산을 하면 출력 데이터(2차원)이 산출 (dot product 한번만 실시하면 됨)
             - 이를 reshape하여 출력 데이터를 만든다.
         - pooling 단계
             - 입력데이터의 한영역을 한줄로 만드는 것은 동일 하지만, 채널들을 row로 concat 시키고 풀링 실시(max or mean)
             - 이를 다시 reshape을 하면 완성
             - 순서
                 - 1. 입력 데이터를 전개
                 - 2. 행력 최대값을 구한다.
                 - 3. 적절한 형태로 reshape 실시
      - 층 깊이에 따른 추출 정보 변화
          - 계층이 깊어질수록 추상화된다 (보고 파악을 못한다는 뜻, 잘되니 그냥 어떤 뜻이 있는갑다 정도)
      - Data augumentation : 인위적으로 확장
        - 이미지를 회전/세로로 이동/미세한 변화로 이미지의 개수 늘리기
        - crop: 이미지 일부를 잘라내기
        - flip: 좌우 반전 (좌우반전이 의미 없는 이미지에 적용)
        - 밝기 등 외형 변화나 확대/축소 등의 스케일 변화도 효과적
---
- Literature Review (to be update...)
    - 1) LeNet (1998)
        - 손글씨 숫자를 인식하는 네트워크
        - sigmoid 사용
            - 현재는 ReLU
        - 풀링 계층: 단순히 원소를 줄이기만하는 서브샘플링 계층
            - 현재는 max pooling
    - 2) AlexNet (2012)
        - 합성곱계층과 풀링을 거듭하여 마지막에 FC
            - ReLU, max pooling, drop out + 병렬연산, GPU
    - 3) VGG (2014)
        - 합성곱 계층, 완전연결 계층 을 모두 16층 (혹은 19층)으로 심화한 게 특징
            - 깊이에 따라 VGG16, VGG19로 표현
        - 3x3의 작은 필터를 사용 / 2~4회 연속으로 풀링 계층을 두어 크기를 절반으로 줄이는 처리를 반복
    - 4) GoogLeNet (2014)
        - INCEPTION 구조
        - 1x1 합성곱을 통해 chanel 수를 줄인다 (1,H,W)로 만든다. 
            - 매개변수 제거와 고속 처리에 기여한다.
    - 5) ResNet (2015)
        - from MS
        - 지나치게 층이 깊으면 학습이 잘 되지 않고, 오히려 성능이 떨어지는 경우도 많다.
        - skip connection 도입
            - 이 구조가 층의 깊이에 비례해 성능을 향상시킬 수 있게 한 핵심
            - 입력 데이터를 계층을 건너뛰어 출력에 바로 더하는 구조
            ![img](img/Resnet.png)
            - 두 합성곱 계층을 건너 뛰어 출력에 바로 연결 F(x)+x가 되는 게 핵심
            - 층이 깊어져도 학습을 효율적으로 할 수 있게 해주며, 역전파 시 신호 감쇠를 막아준다.
            - 스킵 연결로 기울기가 작아지거나 지나치게 커질 걱정 없이 앞 층에 '의미 있는 기울기'가 전해지리라 기대 가능. gradient vanishing 해결
            - 기존 VGG의 구조에 해당 방법을 더해서 진행
            - transfer learning 학습된 가중치(혹은 그 일부)를 다른 신경망에 복사한 다음 그 상태로 재학습을 수행. VGG에서 pretrained 가중치로 초기값을 넣고 새로운 데이터셋을 대상으로 fine tunning을 수행.
                - 데이터 수가 적을 때 유리
---
### Natural Language Process 
---
- 우리가 평소에 말하는 언어를 컴퓨터가 이해하도록 만드는 기술


























---
#### Application
- 사물 검출
    - 이미지속에 담긴 사물의 위치와 종류를 알아내는 기술
    - Regions with Convolutional Neural Network (R-CNN)
        - 최근에 Faster R-CNN 등장
    - Process: 후보 영역 추출 - CNN 특징 계산 - 영역 분류
- 분할 (segmentation)
    - 이미지를 픽셀 수준에서 분류하는 문제
    - 픽셀 단위로 객체마다 채색된 supervised data를 사용해 학습
    - inference 시 입력 이미지의 모든 분류를 분류
        - 각각 픽셀 수만큼 forward를 해야해서 시간이 많이 걸림
        - 이를 해결한 기법이 Fully Convolutional Network (FCN): 한번의 forward로 모든 픽셀의 클래스를 분류해주는 기법 / '합성곱 계층으로만 구성된 네트워크' (기존 CNN의 마지막 레이어를 fully connected가아닌 convolutional을 사용하여 공간 볼륨을 유지한 채 마지막 출력 가능.
        - 마지막에 확대를 실시하는데 이는 bilinear interpolation에 의한 선형 확대.
- 사진 캡션 생성
    - 사진을 주면 그 사진을 설명하는 글(사진 캡션)을 자동으로 생성하는 연구
    - Neural Image Caption
        - depp CNNrhk RNN으로 구성
    - Process: 사진에서 특징 추출 (CNN) - CNN이 추출한 특징을 초기값으로 해서 텍스트를 '순환적'으로 생성 (RNN)
    - multimodal processing: 사진이나 자연어와 같은 여러 종류의 정보를 조합하고 처리하는 것
- 화풍 변화
    - 콘텐츠 이미지 + 스타일 이미지를 조합하여 새로운 그림을 그림
- Image Generation
    - Deep Convolutional Generative Adversial Network (DCGAN)
    - generator는 저 진짜 같은 가짜를, discriminator느느 더 정확히 식별할 수 있도록 학습
- 자율 주행
    - SegNet: CNN 기반 신경망은 주변 환경을 정확하게 인식함.
- 강화학습 (reinforcement learning)
    - agent가 environment에 맞게 action을 선택하고, 그 action에 의해서 environment가 변한다는 게 기본적인 툴
    - environment가 변하면 reward를 얻음. reward를 최대화 시키기 위한 행동지침을 설정
    - Deep Q-Network (DQN): Q학습은 최적 행동 가치 함수로 최적의 행동을 전한다. 이 함수를 CNN으로 비슷하게 흉내내서 사용한 것이 DQN.
- 신약 개발
- 실시간 번역
- 음성 인식
---
    
    