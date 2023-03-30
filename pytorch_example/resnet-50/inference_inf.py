import torch
from torchvision import models
import torch_neuron
from time import time
import numpy as np

image = torch.zeros([1, 3, 224, 224], dtype=torch.float32)
model = models.resnet50(pretrained=True)
model.eval()

model_neuron = torch.neuron.trace(model, example_inputs=[image])

latency = []
throughput = []
# Run inference for 100 iterations and calculate metrics
num_infers = 100000
for _ in range(num_infers):
    delta_start = time()
    results = model_neuron(image)
    delta = time() - delta_start
    latency.append(delta)
    throughput.append(image.size(0)/delta)
    
## 처리 속도 평균
mean_latency = np.mean(latency)
## 1초 / 처리 속도 = 초당 처리 할수있는 이미지 수
throughput = 1 / mean_latency
print(f"1장\t{mean_latency}\t{throughput} 처리")