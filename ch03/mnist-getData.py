import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist

# 데이터셋 로드
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# 데이터 형상 출력
print(f"훈련 이미지 형상: {x_train.shape}")  # (60000, 784)
print(f"훈련 레이블 형상: {t_train.shape}")  # (60000,)
print(f"테스트 이미지 형상: {x_test.shape}")  # (10000, 784)
print(f"테스트 레이블 형상: {t_test.shape}")  # (10000,)
