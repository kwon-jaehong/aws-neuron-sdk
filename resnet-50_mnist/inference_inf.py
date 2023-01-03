import torch
from torchvision import models, transforms, datasets
import torch_neuron
import json
import os
from urllib import request
from time import time
import numpy as np

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
        print("1장 처리시간",delta)
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
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    # Resize the sample image to [1, 3, 224, 224], normalize it, and turn it into a tensor
    eval_dataset = datasets.ImageFolder(
        os.path.dirname("./torch_neuron_test/"),
        transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        normalize,
        ])
    )
    image, _ = eval_dataset[0]
    image = torch.tensor(image.numpy()[np.newaxis, ...])

    # Create a "batched" image with enough images to go on each of the available NeuronCores
    # batch_size is the per-core batch size
    # num_neuron_cores is the number of NeuronCores being used
    batch_image = image
    for i in range(batch_size * num_neuron_cores - 1):
        batch_image = torch.cat([batch_image, image], 0)
     
    return batch_image

image = preprocess(batch_size=batch_size, num_neuron_cores=num_neuron_cores)

# Benchmark the model
benchmark(model_neuron_parallel, image)
