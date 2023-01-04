# 원본 소스
# https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/examples/pytorch/resnet50.ipynb
import torch
from torchvision import models
import torch_neuron
from time import time
import numpy as np

# Run inference with dynamic batch sizes
# Batch size has a direct impact on model performance. The Inferentia chip is optimized to run with small batch sizes. This means that a Neuron compiled model can outperform a GPU model, even if running single digit batch sizes.
# As a general best practice, we recommend optimizing your model's throughput by compiling the model with a small batch size and gradually increasing it to find the peak throughput on Inferentia.
# Dynamic batching is a feature that allows you to use tensor batch sizes that the Neuron model was not originally compiled against. This is necessary because the underlying Inferentia hardware will always execute inferences with the batch size used during compilation. Fixed batch size execution allows tuning the input batch size for optimal performance. For example, batch size 1 may be best suited for an ultra-low latency on-demand inference application, while batch size > 1 can be used to maximize throughput for offline inferencing. Dynamic batching is implemented by slicing large input tensors into chunks that match the batch size used during the torch.neuron.trace compilation call.
# The torch.neuron.DataParallel class automatically enables dynamic batching on eligible models. This allows us to run inference in applications that have inputs with a variable batch size without needing to recompile the model.
# In the following example, we use the same torch.neuron.DataParallel module to run inference using several different batch sizes. Notice that latency increases consistently as the batch size increases. Throughput increases as well, up until a certain point where the input size becomes too large to be efficient.

# 동적 배치 크기로 추론 실행
# 배치 크기는 모델 성능에 직접적인 영향을 미칩니다. Inferentia 칩은 작은 배치 크기로 실행하도록 최적화되어 있습니다. 
# 이는 Neuron 컴파일 모델이 한 자릿수 배치 크기를 실행하는 경우에도 GPU 모델을 능가할 수 있음을 의미합니다.
# 일반적인 모범 사례로 작은 배치 크기로 모델을 컴파일하고 Inferentia에서 최대 처리량을 찾을 수 있도록 점진적으로 증가시켜 
# 모델의 처리량을 최적화하는 것이 좋습니다.
# 동적 배칭은 Neuron 모델이 원래 컴파일되지 않은 텐서 배치 크기를 사용할 수 있게 해주는 기능입니다. 
# 이는 기본 Inferentia 하드웨어가 항상 컴파일 중에 사용된 배치 크기로 추론을 실행하기 때문에 필요합니다. 
# 고정 배치 크기 실행을 통해 최적의 성능을 위해 입력 배치 크기를 조정할 수 있습니다.
# 예를 들어 배치 크기 1은 대기 시간이 매우 짧은 온디맨드 추론 애플리케이션에 가장 적합할 수 있는 반면 배치 크기 > 1은 
# 오프라인 추론을 위한 처리량을 최대화하는 데 사용할 수 있습니다. 
# 동적 일괄 처리는 큰 입력 텐서를 torch.neuron.trace 컴파일 호출 중에 사용된 일괄 처리 크기와 일치하는 청크로 분할하여 구현됩니다.
# torch.neuron.DataParallel 클래스는 적합한 모델에서 자동으로 동적 일괄 처리를 활성화합니다.
# 이를 통해 모델을 다시 컴파일할 필요 없이 가변 배치 크기의 입력이 있는 애플리케이션에서 추론을 실행할 수 있습니다.
# 다음 예에서는 동일한 torch.neuron.DataParallel 모듈을 사용하여 여러 배치 크기를 사용하여 추론을 실행합니다.
# 배치 크기가 증가함에 따라 대기 시간이 지속적으로 증가합니다. 입력 크기가 너무 커서 효율적이지 않은 특정 지점까지 처리량도 증가합니다.


## 배치가 4일때 실험 시작
## aws inf 인스턴스의 뉴런 코어갯수, inf1.xlarge 기준으로 뉴런코어 4개 지원
neuron_core = 4
batch_size = neuron_core
image = torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32)

model = models.resnet50(pretrained=True)
model.eval()
## aws neuron sdk에서 컨버팅 지원되는지 확인하는 구문
torch.neuron.analyze_model(model, example_inputs=[image])

model_neuron = torch.neuron.trace(model, example_inputs=[image])
model_neuron.save("resnet50_neuron.pt")
## AWS neuron 뉴런 모델 세이브

# Load the compiled Neuron model
model_neuron = torch.jit.load('resnet50_neuron.pt')
model_neuron_parallel = torch.neuron.DataParallel(model_neuron)

latency = []
throughput = []

## 총 100장을 추론하려고 함
## 배치가 4니까 반복문은 25번 돔
num_infers = int(100 / neuron_core)
for _ in range(num_infers):
    delta_start = time()
    results = model_neuron_parallel(image)
    delta = time() - delta_start
    latency.append(delta)

total_latency = np.sum(latency)   
## 1초 / 처리 속도 = 초당 처리 할수있는 이미지 수
print(f"배치가 4일때 컴파일 모델 \t{total_latency/100}\t{1/(total_latency/100)} 처리")








