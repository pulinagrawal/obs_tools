from transformers import ViTFeatureExtractor, ViTModel, ViTForImageClassification
from transformers import AutoFeatureExtractor, CvtForImageClassification, ResNetForImageClassification
from itertools import combinations, product
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import requests
import numpy as np

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/cvt-13')
model = CvtForImageClassification.from_pretrained('microsoft/cvt-13')
model.config.output_hidden_states = True
# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-huge-patch14-224-in21k')
# model = ViTModel.from_pretrained('google/vit-huge-patch14-224-in21k')
# inputs = feature_extractor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# last_hidden_states = outputs.last_hidden_state

# program to capture single image from webcam in python
#find cv2 image to PIL image.
def toImgOpenCV(imgPIL): # Conver imgPIL to imgOpenCV
    i = np.array(imgPIL) # After mapping from PIL to numpy : [R,G,B,A]
                         # numpy Image Channel system: [B,G,R,A]
    red = i[:,:,0].copy(); i[:,:,0] = i[:,:,2].copy(); i[:,:,2] = red;
    return i; 

def toImgPIL(imgOpenCV): return Image.fromarray(cv2.cvtColor(imgOpenCV, cv2.COLOR_BGR2RGB));

def run_on_folder():
    data_folder = Path('data/natural_images')
    data = {}
    label_folder = ''
    reps = {}
    for lcount, label_folder in tqdm(enumerate(data_folder.iterdir()), desc=f'folder'):
        if lcount>3:
            break
        data[label_folder.name] = []
        reps[label_folder.name] = []
        print(label_folder.name)
        for count, file in tqdm(enumerate(label_folder.iterdir())):
            image = Image.open(file)
            inputs = feature_extractor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            # model predicts one of the 1000 ImageNet classes
            predicted_class_idx = logits.argmax(-1).item()
            print("Predicted class: ", model.config.id2label[predicted_class_idx])
            reps[label_folder.name].append(outputs.hidden_states[-1].detach())
            if count>30:
                break
            
    length = reps[list(reps.keys())[0]][0].flatten().shape[0]
    dist = []
    for p1, p2 in combinations(reps[list(reps.keys())[0]], 2):
        dist.append((p1.flatten()-p2.flatten()).pow(2).sum().item()/length) 

    df = pd.DataFrame(dist)
    print(list(reps.keys())[0])
    print(df.describe())

    for k1, k2 in combinations(reps, 2):
        dist = []
        for pair in product(reps[k1], reps[k2]):
           dist.append((pair[0].flatten()-pair[1].flatten()).pow(2).sum().item()/length) 
        df = pd.DataFrame(dist)
        print(k1, k2)
        print(df.describe())

run_on_folder()
    
def run_on_webcam():
    import cv2

    cam_port = 1
    cam = cv2.VideoCapture(cam_port)
    while True:

        result, image = cam.read()
        cv2.imshow("GeeksForGeeks", image)

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyWindow("GeeksForGeeks")
            break
        
        pil_image = toImgPIL(image)
        inputs = feature_extractor(images=pil_image, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        print(last_hidden_states)