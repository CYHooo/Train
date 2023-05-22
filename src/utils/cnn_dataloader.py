import os 
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from ultralytics import YOLO


class TorchData():
    def __init__(self,data_dir, labels, img_size, random_state, batch_size) -> None:
        self.data_dir     = data_dir
        self.labels       = labels
        self.img_size     = img_size
        self.random_state = random_state
        self.batch_size   = batch_size

    def read_images(self):
        data = []
        for label in self.labels: 
            path = os.path.join(self.data_dir, label)
            class_num = self.labels.index(label)
            for img in os.listdir(path):
                img_arr = cv2.imread(os.path.join(path, img))
                resized_arr = cv2.resize(img_arr, (self.img_size, self.img_size)) 
                data.append([resized_arr, class_num])
        # self.dataset = np.array(data, dtype=object)
        return np.array(data, dtype=object)
    
    def train_data(self, data):
        x,y = [], []

        for feature, label in data:
            x.append(feature)
            y.append(label)
        x = np.array(x).reshape(-1, self.img_size, self.img_size, 3)
        x = x/255
        x = np.transpose(x,(0,3,1,2))
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=self.random_state)

        ## 'np.array' data to 'torch.tensor' data
        X_train = torch.tensor(X_train).float()
        y_train = torch.tensor(y_train).long()
        X_test = torch.tensor(X_test).float()
        y_test = torch.tensor(y_test).long()

        ## Pytorch Tensor Dataloader
        train_data = TensorDataset(X_train, y_train)
        test_data = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)

        return train_loader, test_loader
    
    def test_data(self,data):
        x = []
        # y = []
        for feature in data:
            feature = np.resize(feature,(self.img_size,self.img_size,3))
            x.append(feature)
            # y.append(label)
        x = np.array(x,dtype=float).reshape(-1, self.img_size, self.img_size, 3)
        x = x/255
        x = np.transpose(x,(0,3,1,2))
        # y = np.array(y)

        tensor_img = torch.tensor(x).float()
        # tensor_label = torch.tensor(y).long()

        # Tensor_data = TensorDataset(tensor_img, tensor_label)
        Tensor_data = TensorDataset(tensor_img)

        data_loder = DataLoader(Tensor_data)

        return data_loder

class YoloData():
    def __init__(self, img_path: os.path, weight: str) -> None:
        self.img_path   = img_path
        self.weight     = weight

    def result(self):
        box, crop = [], []
        img = cv2.imread(self.img_path)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        model = YOLO(self.weight)

        predicts = model.predict(source=img)
        for predict in predicts:
            index = predict.boxes.data
            bboxes = index[:,:4].tolist()
            for bbox in bboxes:
            # label = predict.names[index[0,5].tolist()]
                crop_img = img[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

                box.append(bbox)
                crop.append(crop_img)

        return box, crop
