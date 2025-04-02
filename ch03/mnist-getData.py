import os
import sys

# 현재 파일의 절대 경로를 기준으로 부모 디렉토리 설정
current = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from dataset.mnist import load_mnist
from PIL import Image
import numpy as np

# 데이터셋 로드
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# # 데이터 형상 출력
# print(f"훈련 이미지 형상: {x_train.shape}")  # (60000, 784)
# print(f"훈련 레이블 형상: {t_train.shape}")  # (60000,)
# print(f"테스트 이미지 형상: {x_test.shape}")  # (10000, 784)
# print(f"테스트 레이블 형상: {t_test.shape}")  # (10000,)


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 원래 이미지의 모양으로 변형
print(img.shape)  # (28, 28)

img_show(img)
