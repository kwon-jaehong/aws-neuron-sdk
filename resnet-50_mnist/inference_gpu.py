import torch
from torchvision import models
from torch.utils.data import DataLoader,dataset
from torchsummary import summary as summary_
from time import time

USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("device :", device)


# Load a pretrained ResNet50 model
model = models.resnet50(pretrained=True)
model.to(device=device)
model.eval()
summary_(model,(3, 224, 224),batch_size=10)

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
image = torch.zeros([1, 3, 224, 224], dtype=torch.float32)
# image = preprocess(batch_size=batch_size, num_neuron_cores=4)

# Benchmark the model
benchmark(model, image)
