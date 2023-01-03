import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuron

# Create an example input for compilation
image = torch.zeros([1,1, 28, 28], dtype=torch.float32)

# Load a pretrained ResNet50 model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding='same')
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding='same')
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
        output = F.log_softmax(x, dim=1)
        return output


model = CNN()
# Tell the model we are using it for evaluation (not training)
model.eval()

# Analyze the model - this will show operator support and operator count
## aws 뉴런 sdk로 인풋 이미지를 직접 forward시키면서 분석
torch.neuron.analyze_model(model, example_inputs=[image])


# Compile the model using torch.neuron.trace to create a Neuron model
# that that is optimized for the Inferentia hardware
# aws 뉴런 sdk를 이용해서 inf 인스턴스(하드웨어)에 최적화된 구조체 생성
model_neuron = torch.neuron.trace(model, example_inputs=[image])


# The output of the compilation step will report the percentage of operators that
# are compiled to Neuron, for example:
#
# INFO:Neuron:The neuron partitioner created 1 sub-graphs
# INFO:Neuron:Neuron successfully compiled 1 sub-graphs, Total fused subgraphs = 1, Percent of model sub-graphs successfully compiled = 100.0%
#
# We will also be warned if there are operators that are not placed on the Inferentia hardware

# Save the compiled model
model_neuron.save("inf1_mnist.pt")