from ultralytics import YOLO
import torch
from src.utils.frcnn_training import fast_rcnn_training

def train_objectdetection(config, label):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dir_data, dir_label, model, epochs, lr, batch_size, optimizer = config['dir_data'],\
                                                                    config['dir_label'],\
                                                                    config['model'],\
                                                                    config['epochs'],\
                                                                    config['lr'],\
                                                                    config['batch_size'],\
                                                                    config['optimizer']


    if optimizer =='adam':
        optimizer = 'Adam'
    elif optimizer == "sgd":
        optimizer = 'SGD'

    if device.type == "cuda":
        device = 0
    else:
        device = "cpu"

    if model.lower() == 'yolov8s':
        model = YOLO('yolov8s.pt')
        model.train(data=dir_data, epochs=epochs, batch=batch_size, imgsz=1024, device=device, optimizer=optimizer)
    elif model.lower() == 'faster r-cnn':
        fast_rcnn_training(model, dir_label, label, epochs, lr, dir_data, optimizer)

    

