import torch
from torchvision import models
import torch_neuron
from time import time
import numpy as np
import os
from fastapi import FastAPI,Form,UploadFile
import uvicorn
from typing import Union
from io import BytesIO
import base64
from PIL import Image
from torchvision import transforms


## 사용가능 코어를 다시 지정 해줘야함 -> 오류 찾는데 하루 걸림 0번째 코어가 자동적으로 배치가 안되는 문제가 있엇음
os.environ['NEURON_RT_VISIBLE_CORES'] = str(os.environ['NEURON_RT_VISIBLE_CORES'])



transform = transforms.Compose([
    # 0~1의 범위를 가지도록 정규화
    transforms.ToTensor(),
    transforms.Resize((224,224))
])

# image = torch.zeros([1, 3, 224, 224], dtype=torch.float32)
if os.path.isfile(os.path.join(os.getcwd(),'resnet50_neuron.pt')):
    model_neuron = torch.jit.load(os.path.join(os.getcwd(),'resnet50_neuron.pt'))
else:
    model = models.resnet50(pretrained=True)
    model.eval()
    model_neuron = torch.neuron.trace(model, example_inputs=[image])
    model_neuron.save(os.path.join(os.getcwd(),'resnet50_neuron.pt'))

app = FastAPI()
@app.post("/resnet")
async def upload(filename: str = Form(...), filedata: Union[str, UploadFile] = Form(...)):
    try:
        if "UploadFile" in str(type(filedata)):
            print("파일임",filedata.filename)
        
        # base64기반 파일
        elif type(filedata) == str:
            img = Image.open(BytesIO(base64.b64decode(filedata))).convert('RGB')
            img = transform(img).unsqueeze(0)
            result = model_neuron(img)

        else:
            raise
    except:
        print("받은 파일 알수 없음")
    # results = model_neuron(image)
    return {"result":str(result)}



if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)