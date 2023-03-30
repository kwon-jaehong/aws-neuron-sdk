import torch
from torchvision import models
import torch_neuron
from time import time
import numpy as np
import os


## 분산할 뉴런 코어 갯수
num_cores = 4 


image = torch.zeros([1, 3, 224, 224], dtype=torch.float32)
model = models.resnet50(pretrained=True)
model.eval()

## 뉴런코어 4개로 분산하여 컴파일하겠다
model_neuron = torch.neuron.trace(model, example_inputs=[image],compiler_args = ['--neuroncore-pipeline-cores', str(num_cores)])


latency = []


num_infers = 10000
for _ in range(num_infers):
    delta_start = time()
    results = model_neuron(image)
    delta = time() - delta_start
    latency.append(delta)
    
## 처리 속도 평균
mean_latency = np.mean(latency)
## 1초 / 처리 속도 = 초당 처리 할수있는 이미지 수
throughput = 1 / mean_latency
print(f"1장\t{mean_latency}\t{throughput} 처리")