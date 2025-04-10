import os
import sys
import time
from datetime import datetime

# 현재 파일의 절대 경로를 기준으로 부모 디렉토리 설정
current = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt


def print_progress(
    iteration,
    total,
    loss,
    accuracy,
    start_time,
    prefix="",
    suffix="",
    decimals=1,
    bar_length=50,
):
    """
    학습 진행률을 보여주는 프로그레스 바
    """
    filled_length = int(round(bar_length * iteration / float(total)))
    percents = round(100.0 * iteration / float(total), decimals)
    elapsed_time = time.time() - start_time
    estimated_total = elapsed_time / (iteration + 1) * total if iteration > 0 else 0
    remaining_time = estimated_total - elapsed_time

    # 시간 형식 변환
    elapsed = str(datetime.utcfromtimestamp(int(elapsed_time)).strftime("%H:%M:%S"))
    remaining = str(datetime.utcfromtimestamp(int(remaining_time)).strftime("%H:%M:%S"))

    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    print(
        f"\r{prefix} |{bar}| {percents}% {suffix} Loss: {loss:.4f} Acc: {accuracy:.2f}% ET: {elapsed} ETA: {remaining}",
        end="",
    )
    if iteration == total:
        print()


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 하이퍼파라미터
iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
start_time = time.time()

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch)
    # grad = network.gradient(x_batch, t_batch) # 성능 개선판!

    # 매개변수 갱신
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 정확도 계산
    train_acc = network.accuracy(x_batch, t_batch)
    if i % 100 == 0:  # 100번째 반복마다 테스트 데이터로 정확도 계산
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

    # 진행상황 출력
    print_progress(
        i + 1,
        iters_num,
        loss,
        train_acc * 100,
        start_time,
        prefix="Progress:",
        suffix="Complete",
    )

# 그래프 그리기
plt.figure(figsize=(15, 5))

# Loss 그래프
plt.subplot(1, 2, 1)
x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list, label="loss")
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 3)
plt.legend()

# Accuracy 그래프
plt.subplot(1, 2, 2)
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label="train acc")
plt.plot(x, test_acc_list, label="test acc", linestyle="--")
plt.xlabel("iterations (x100)")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend()

plt.show()
