import torch
from torchvision import models
from time import time
import numpy as np

USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("device :", device)

# Load a pretrained ResNet50 model
model = models.resnet50(pretrained=True)
model.to(device=device)
model.eval()


# print(f"모델 파라미터 갯수 {sum(p.numel() for p in model.parameters())}")



image = torch.zeros([1, 3, 224, 224], dtype=torch.float32)
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