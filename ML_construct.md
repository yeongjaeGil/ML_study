## model construct
---
- confis : 실험 세팅
- etc : 이미지 같은 것들
- ML_name : 모델 관련 파일
    - network
    - preprocessing
        - 1. data load/pretrained 호출 : download.sh
        - 2. 데이터 전처리/pickle로 : preprocessing.py
        - 3. pytorch용 전처리 : dataset.py
            - Dataset
            - Transforms
            - DataLoader
        - * 2, 3번을 묶어서 data_loader.py로 만들기
    - model.py로 따로 작성하여 재사용성과 가독성을 높인다.
    - train.py (학습용)
        - 하이퍼파라미터 설정
            - 1. config.yml / yaml 라이브러리 사용
            - 2. Argparse 라이브러리 사용해서 cmd로 인자값 받기 -> .sh 파일로 구성
    - test.py (sample.py/inference.py)로 구성
        - Logger.py
            - Tensorboard/Visdom
     - utils.py로 나머지 저장
     - main.py
         
    
    functions
