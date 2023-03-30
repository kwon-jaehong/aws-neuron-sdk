from time import time
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch


USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("device :", device)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.dropout = nn.Dropout2d(0.25)
        # (입력 뉴런, 출력 뉴런)
        self.fc1 = nn.Linear(3136, 1000)    # 7 * 7 * 64 = 3136
        self.fc2 = nn.Linear(1000, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

# Load a pretrained ResNet50 model
model = CNN()
model.to(device=device)
model.eval()


print(f"모델 파라미터 갯수 {sum(p.numel() for p in model.parameters())}")


image = torch.zeros([1, 1, 28, 28], dtype=torch.float32)
latency = []
image = image.to(device=device)
# 1장씩 100번 넣음
num_infers = 100000
for _ in range(num_infers):
    delta_start = time()
    results = model(image)
    delta = time() - delta_start
    latency.append(delta)

# 100번 넣은 반응속도 리스트
# print(latency)

## 처리 속도 평균
mean_latency = np.mean(latency)

## 1초 / 처리 속도 = 초당 처리 할수있는 이미지 수
throughput = 1 / mean_latency
print(f"1장\t{mean_latency}\t{throughput} 처리")