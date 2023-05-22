import os
import numpy as np
import cv2
from PIL import Image
from matplotlib.pyplot import plot as plt

import torch
from utils.unet_dataloader import TorchData, YoloData
from utils.training import training

from src.nets.unet import Unet


class processing():
    def __init__(self, 
                    images: list, 
                    yolo_weight: str,
                    cnn_model: torch.ModuleDict,
                    device,
                 ) -> None:
        
        self.images          = images
        self.yolo_weight     = yolo_weight
        self.cnn_model       = cnn_model
        self.device          = device
        

    def preprocessing(self) -> list:
        preprocess = []
        for img in self.images:
            infos = []
            
            yolo = YoloData(img, self.yolo_weight)
            boxes, crops = yolo.result()

            cnn = TorchData.test_data(np.array(crops, dtype=object))
            predicts = training.test(model=self.cnn_model, test_loader=cnn, device=self.device)

            for box, predict in zip(boxes, predicts):
                box.extend(predict.tolist())
                infos.append(box)  ## preprocess: list[[x1,y1,x2,y2,class.index], [...], ...]
            preprocess.append(infos)
        return preprocess

    def postprocessing(self, infos: list, save_path: os.path, label: list) -> list:
        self.blend = []
        print('Saving image ...')
        unet = Unet()
        color = [(0,0,255),(0,255,0)]
        for img, pre_info in zip(self.images, infos):
            img_name = os.path.basename(img).rsplit('.', 1)[0]
            road_mask = unet.detect_image(img)
            print('[Image Input]:', f'{os.path.abspath(img)}')
            img = cv2.imread(img)
            for info in pre_info:
                x1, y1, x2, y2, index = info
                cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), color[index], 3)
                # cv2.putText(img, label[index], (int(x1),int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color[index], 2)
            result = cv2.addWeighted(road_mask, 0.5, img, 0.7, gamma=0)
            cv2.imwrite(save_path + img_name + '_result.png', result)
            print('[Image Save in Path]:', f'{save_path}{img_name}_result.png\n')
        self.blend.append(result)
        print('\nDone!!')
        return self.blend
    
    # def show(self):
    #     for img in self.blend:

