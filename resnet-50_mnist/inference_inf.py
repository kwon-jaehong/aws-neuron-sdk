import torch
from torchvision import models, transforms, datasets
import torch_neuron

# Create an example input for compilation
image = torch.zeros([1, 3, 224, 224], dtype=torch.float32)

# Load a pretrained ResNet50 model
model = models.resnet50(pretrained=True)

# Tell the model we are using it for evaluation (not training)
model.eval()

# Analyze the model - this will show operator support and operator count
torch.neuron.analyze_model(model, example_inputs=[image])

# Compile the model using torch.neuron.trace to create a Neuron model
# that that is optimized for the Inferentia hardware
model_neuron = torch.neuron.trace(model, example_inputs=[image])

# The output of the compilation step will report the percentage of operators that 
# are compiled to Neuron, for example:
#
# INFO:Neuron:The neuron partitioner created 1 sub-graphs
# INFO:Neuron:Neuron successfully compiled 1 sub-graphs, Total fused subgraphs = 1, Percent of model sub-graphs successfully compiled = 100.0%
# 
# We will also be warned if there are operators that are not placed on the Inferentia hardware

# Save the compiled model
model_neuron.save("resnet50_neuron.pt")

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
    
import numpy as np

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
import torch
from torchvision import models, transforms, datasets
import torch_neuron

# Get a sample image
image = preprocess()

# Run inference using the CPU model
output_cpu = model(image)

# Load the compiled Neuron model
model_neuron = torch.jit.load('resnet50_neuron.pt')

# Run inference using the Neuron model
output_neuron = model_neuron(image)

# Verify that the CPU and Neuron predictions are the same by comparing
# the top-5 results
top5_cpu = output_cpu[0].sort()[1][-5:]
top5_neuron = output_neuron[0].sort()[1][-5:]

# Lookup and print the top-5 labels
top5_labels_cpu = [idx2label[idx] for idx in top5_cpu]
top5_labels_neuron = [idx2label[idx] for idx in top5_neuron]
print("CPU top-5 labels: {}".format(top5_labels_cpu))
print("Neuron top-5 labels: {}".format(top5_labels_neuron))


from time import time

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
    
    # Calculate and print the model throughput and latency
    print("Avg. Throughput: {:.0f}, Max Throughput: {:.0f}".format(np.mean(throughput), np.max(throughput)))
    print("Latency P50: {:.0f}".format(np.percentile(latency, 50)*1000.0))
    print("Latency P90: {:.0f}".format(np.percentile(latency, 90)*1000.0))
    print("Latency P95: {:.0f}".format(np.percentile(latency, 95)*1000.0))
    print("Latency P99: {:.0f}\n".format(np.percentile(latency, 99)*1000.0))
    
    
    # Create a torch.neuron.DataParallel module using the compiled Neuron model
# By default, torch.neuron.DataParallel will use four cores on an inf1.xlarge
# or inf1.2xlarge, 16 cores on an inf1.6xlarge, and 24 cores on an inf1.24xlarge
model_neuron_parallel = torch.neuron.DataParallel(model_neuron)

# Get sample image with batch size=1 per NeuronCore
batch_size = 1

# For an inf1.xlarge or inf1.2xlarge, set num_neuron_cores = 4
num_neuron_cores = 16

image = preprocess(batch_size=batch_size, num_neuron_cores=num_neuron_cores)

# Benchmark the model
benchmark(model_neuron_parallel, image)

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

