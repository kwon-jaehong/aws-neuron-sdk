import base64
import requests

# url = "http://example-app.default.svc:8000/resnet"
url = 'http://localhost:8000/resnet'
with open("./image.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
    
# payload ={"filename": "photo.png"}

payload ={"filename": "photo.png", "filedata": encoded_string}

for i in range(0,100000):
    resp = requests.post(url=url, data=payload) 
    print(resp)
# print(resp.json())