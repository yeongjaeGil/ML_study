#### torch 특징
- Python First
- numpy와 호환성
- Autograd
- 동적 그래프: 연산과 종시에 동적 그래프가 생기므로 자유로움

#### logging
- 로깅은 어떤 소프트웨어가 실행될 때 발생하는 이벤트를 추적하는 수단
- 소프트웨어 개발자는 코드에 로깅 호출을 추가하여 특정 이벤트가 발생했음을 나타냄
- 이벤트는 또한 개발자가 이벤트에 부여한 중요도를 가지고 있음.
    - level/severity 라고 부름
    - logging.info() : 프로그램의 정상 작동 중에 발생하는 이벤트 보고 (상태 모니터링이나 결함 조사)
    - logging.debug() : 상세한 정보, 보통 문제를 진단할 때만 필요
    - logging.warning() : 특정 실행시간 이벤트와 관련하여 경고를 발행, 클라이언트 응용 프로그램이 할 수 있는 일이 없는 상황이지만 이벤트를 계속 주목해야 하는 경우
    - logging.error(), logging,exception(), logging.critical(): 예외를 발생시키지 않고 에러의 억제를 보고 (가령 장기 실행 서버 프로세스의 에러 처리기)
    - logging.basicConfig(format='%(asctime)s %(message)s') : 기본적으로 날짜 표시
    - *logging을 통해서 loss를 저장해야함.
    - torch.save() / torch.load()로 관리해야함.


---
#### 개요
- 파이토치는 성능적 이유로 대부분 C++과 CUDA언어로 만들어짐. NVIDIA의 GPU를 통한 병렬처리가 가능함.
- 파이토치 모델은 Flask 웹 서버를 사용하여 실행될 수 있다.
- Dataset: custom 데이터를 pytorch 표준 텐서로 바꿀 수 있음
- Transforms
    - Rescale
    - RandomCrop: 임의로 이미지를 자른다(data augmentation)
    - ToTensor: numpy 배열의 이미지를 torch 텐서로 바꿔준다.


- DataLoader를 통해 배치단위로 학습 루프에 들어가기 위한 데이터를 준비해주는 데이터 로더를 만들 수 있음
- GPU를 데이터 로딩과 학습 연산에 사용할 수 있도록 torch.nn.DataParallel과 torch.distributed를 지원
- python 인터프리터 비용을 줄이고 python 런타임으로부터 독립적으로 모델을 실행시키기 위해, TorchScript를 지원. 텐서 연산에 제한된 가상머신이라고 생각하면 됨. Just in Time(JIT)을 지원

---
#### Storage
- 텐서는 모양은 다양할 수 있지만, 저장 공간은 1차원 배열로 storage에 저장된다.
- .to(device) 는 같은 숫자 데이터를 갖는 새로운 텐서를 반환하며, 기본 RAM이 아닌 GPU의 RAM안에 저장한다.


---
- train()과 eval()을 구분시켜줘야함. inference 단계는 dropout이나 batch normalization등이 각 상황에 맞게 동작함.

[pytorch](https://wordbe.tistory.com/entry/Pytorch-1-%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98%EB%A5%BC-%EC%8D%A8%EC%95%BC%ED%95%98%EB%8A%94-%EC%9D%B4%EC%9C%A0-%ED%85%90%EC%84%9C%EB%9E%80)
