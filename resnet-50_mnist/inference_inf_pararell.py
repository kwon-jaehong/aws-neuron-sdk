import torch
from torchvision import models, transforms, datasets
import torch_neuron
import json
import os
from urllib import request
from time import time
import numpy as np

import json
import os
from urllib import request

# Create an image directory containing a sample image of a small kitten
os.makedirs("./torch_neuron_test/images", exist_ok=True)
request.urlretrieve("https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg",
                    "./torch_neuron_test/images/kitten_small.jpg")

# Fetch labels to output the top classifications
request.urlretrieve("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json","imagenet_class_index.json")
idx2label = []

# Read the labels and create a list to hold them for classification 
with open("imagenet_class_index.json", "r") as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    

image = torch.zeros([1, 3, 224, 224], dtype=torch.float32)
model = models.resnet50(pretrained=True)
model.eval()

## aws neuron sdk에서 컨버팅 지원되는지 확인하는 구문
torch.neuron.analyze_model(model, example_inputs=[image])

model_neuron = torch.neuron.trace(model, example_inputs=[image])
model_neuron.save("resnet50_neuron.pt")



image = torch.zeros([1, 3, 224, 224], dtype=torch.float32)

# Run inference using the CPU model
output_cpu = model(image)

# Load the compiled Neuron model
model_neuron = torch.jit.load('resnet50_neuron.pt')

def benchmark(model, image):
    print('Input image shape is {}'.format(list(image.shape)))
    
    # The first inference loads the model so exclude it from timing 
    results = model(image)
    
    # Collect throughput and latency metrics
    latency = []
    throughput = []

    # Run inference for 100 iterations and calculate metrics
    num_infers = 100
    for _ in range(num_infers):
        delta_start = time()
        results = model(image)
        delta = time() - delta_start
        latency.append(delta)
        throughput.append(image.size(0)/delta)


model_neuron_parallel = torch.neuron.DataParallel(model_neuron)

# Get sample image with batch size=1 per NeuronCore
# NeuronCore당 샘플 이미지 1장 가져오기
batch_size = 1

# For an inf1.xlarge or inf1.2xlarge, set num_neuron_cores = 4
# inf1.xlarge 또는 inf1.2xlarge의 경우 num_neuron_cores = 4로 설정
num_neuron_cores = 4


def preprocess(batch_size=1, num_neuron_cores=1):
    # Define a normalization function using the ImageNet mean and standard deviation
    # ImageNet 평균과 표준 편차를 사용하여 정규화 함수를 정의합니다.
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    # Resize the sample image to [1, 3, 224, 224], normalize it, and turn it into a tensor
    # 샘플 이미지의 크기를 [1, 3, 224, 224]로 조정하고 정규화한 다음 텐서로 변환
    eval_dataset = datasets.ImageFolder(
        os.path.dirname("./torch_neuron_test/"),
        transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        normalize,
        ])
    )
    
    image, _ = eval_dataset[0]
    # image = torch.Size([3, 224, 224]) torch.float32
    
    image = torch.tensor(image.numpy()[np.newaxis, ...])
    # image = torch.Size([1, 3, 224, 224]) torch.float32

    # Create a "batched" image with enough images to go on each of the available NeuronCores
    # batch_size is the per-core batch size
    # num_neuron_cores is the number of NeuronCores being used
    # 사용 가능한 각 NeuronCore에 들어갈 충분한 이미지로 "일괄 처리된" 이미지를 생성합니다.
    # batch_size는 코어당 배치 크기입니다.
    # num_neuron_cores는 사용 중인 NeuronCore의 수입니다.
    # inf 인스턴스의 뉴런 코어수만큼 이미지 배치를 만들어줌
    batch_image = image
    for i in range(batch_size * num_neuron_cores - 1):
        # batch_size * num_neuron_cores - 1 = 3
        batch_image = torch.cat([batch_image, image], 0)
     
    # batch_image.shape = torch.Size([4, 3, 224, 224])
    return batch_image

image = preprocess(batch_size=batch_size, num_neuron_cores=num_neuron_cores)

# Benchmark the model
benchmark(model_neuron_parallel, image)





# Run inference with dynamic batch sizes
# Batch size has a direct impact on model performance. The Inferentia chip is optimized to run with small batch sizes. This means that a Neuron compiled model can outperform a GPU model, even if running single digit batch sizes.

# As a general best practice, we recommend optimizing your model's throughput by compiling the model with a small batch size and gradually increasing it to find the peak throughput on Inferentia.

# Dynamic batching is a feature that allows you to use tensor batch sizes that the Neuron model was not originally compiled against. This is necessary because the underlying Inferentia hardware will always execute inferences with the batch size used during compilation. Fixed batch size execution allows tuning the input batch size for optimal performance. For example, batch size 1 may be best suited for an ultra-low latency on-demand inference application, while batch size > 1 can be used to maximize throughput for offline inferencing. Dynamic batching is implemented by slicing large input tensors into chunks that match the batch size used during the torch.neuron.trace compilation call.

# The torch.neuron.DataParallel class automatically enables dynamic batching on eligible models. This allows us to run inference in applications that have inputs with a variable batch size without needing to recompile the model.

# In the following example, we use the same torch.neuron.DataParallel module to run inference using several different batch sizes. Notice that latency increases consistently as the batch size increases. Throughput increases as well, up until a certain point where the input size becomes too large to be efficient.

# 동적 배치 크기로 추론 실행
# 배치 크기는 모델 성능에 직접적인 영향을 미칩니다. Inferentia 칩은 작은 배치 크기로 실행하도록 최적화되어 있습니다. 이는 Neuron 컴파일 모델이 한 자릿수 배치 크기를 실행하는 경우에도 GPU 모델을 능가할 수 있음을 의미합니다.

# 일반적인 모범 사례로 작은 배치 크기로 모델을 컴파일하고 Inferentia에서 최대 처리량을 찾을 수 있도록 점진적으로 증가시켜 모델의 처리량을 최적화하는 것이 좋습니다.

# 동적 배칭은 Neuron 모델이 원래 컴파일되지 않은 텐서 배치 크기를 사용할 수 있게 해주는 기능입니다. 이는 기본 Inferentia 하드웨어가 항상 컴파일 중에 사용된 배치 크기로 추론을 실행하기 때문에 필요합니다. 고정 배치 크기 실행을 통해 최적의 성능을 위해 입력 배치 크기를 조정할 수 있습니다. 예를 들어 배치 크기 1은 대기 시간이 매우 짧은 온디맨드 추론 애플리케이션에 가장 적합할 수 있는 반면 배치 크기 > 1은 오프라인 추론을 위한 처리량을 최대화하는 데 사용할 수 있습니다. 동적 일괄 처리는 큰 입력 텐서를 torch.neuron.trace 컴파일 호출 중에 사용된 일괄 처리 크기와 일치하는 청크로 분할하여 구현됩니다.

# torch.neuron.DataParallel 클래스는 적합한 모델에서 자동으로 동적 일괄 처리를 활성화합니다. 이를 통해 모델을 다시 컴파일할 필요 없이 가변 배치 크기의 입력이 있는 애플리케이션에서 추론을 실행할 수 있습니다.

# 다음 예에서는 동일한 torch.neuron.DataParallel 모듈을 사용하여 여러 배치 크기를 사용하여 추론을 실행합니다. 배치 크기가 증가함에 따라 대기 시간이 지속적으로 증가합니다. 입력 크기가 너무 커서 효율적이지 않은 특정 지점까지 처리량도 증가합니다.


batch_sizes = [2, 3, 4, 5, 6, 7]
for batch_size in batch_sizes:
    print('Batch size: {}'.format(batch_size))
    image = preprocess(batch_size=batch_size, num_neuron_cores=num_neuron_cores)
    
    # Benchmark the model for each input batch size
    benchmark(model_neuron_parallel, image)

# Compile and Infer with different batch sizes on multiple NeuronCores
# Dynamic batching using small batch sizes can result in sub-optimal throughput because it involves slicing tensors into chunks and iteratively sending data to the hardware. Using a larger batch size at compilation time can use the Inferentia hardware more efficiently in order to maximize throughput. You can test the tradeoff between individual request latency and total throughput by fine-tuning the input batch size.

# In the following example, we recompile our model using a batch size of 5 and run the model using torch.neuron.DataParallel to fully saturate our Inferentia hardware for optimal performance.
# 여러 NeuronCore에서 다양한 배치 크기로 컴파일 및 추론
# 작은 배치 크기를 사용하는 동적 배치는 텐서를 청크로 분할하고 반복적으로 데이터를 하드웨어로 전송하기 때문에 차선의 처리량을 초래할 수 있습니다. 컴파일 시간에 더 큰 배치 크기를 사용하면 처리량을 최대화하기 위해 Inferentia 하드웨어를 더 효율적으로 사용할 수 있습니다. 입력 일괄 처리 크기를 미세 조정하여 개별 요청 대기 시간과 총 처리량 간의 균형을 테스트할 수 있습니다.

# 다음 예에서는 배치 크기 5를 사용하여 모델을 다시 컴파일하고 최적의 성능을 위해 Inferentia 하드웨어를 완전히 포화시키기 위해 torch.neuron.DataParallel을 사용하여 모델을 실행합니다.

# Create an input with batch size 5 for compilation
batch_size = 5
image = torch.zeros([batch_size, 3, 224, 224], dtype=torch.float32)

# Recompile the ResNet50 model for inference with batch size 5
model_neuron = torch.neuron.trace(model, example_inputs=[image])

# Export to saved model
model_neuron.save("resnet50_neuron_b{}.pt".format(batch_size))


batch_size = 5

# Load compiled Neuron model
model_neuron = torch.jit.load("resnet50_neuron_b{}.pt".format(batch_size))

# Create DataParallel model
model_neuron_parallel = torch.neuron.DataParallel(model_neuron)

# Get sample image with batch size=5
image = preprocess(batch_size=batch_size, num_neuron_cores=num_neuron_cores)

# Benchmark the model
benchmark(model_neuron_parallel, image)






