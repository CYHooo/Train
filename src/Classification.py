## base package
import os 
import numpy as np

## torch package
import torch
import torch.optim as optim
from torch import nn

## model
from src.nets.cnn_model import CNN

## utils package
from src.utils.cnn_dataloader import TorchData
from src.utils.training import training
from src.utils.hyperparameter import hyperparameter
from src.nets.CNN_resnet import Bottleneck, ResNet, ResNet50, ResNet101
from src.nets.alexnet import AlexNet


## check device on 'cpu' or 'gpu'


def train_classification(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dir_data, dir_label, model, epochs, lr, batch_size, optimizer = config['dir_data'],\
                                                                    config['dir_label'],\
                                                                    config['model'],\
                                                                    config['epochs'],\
                                                                    config['lr'],\
                                                                    config['batch_size'],\
                                                                    config['optimizer']
    
    labels = os.listdir(dir_label)
    img_size = 256

    model_name = model
    save_path = f"weight/{model_name}/"

    dataload =  TorchData(dir_data, labels, img_size, random_state=42, batch_size=batch_size)
    data = dataload.read_images()
    
    train_loader,test_loader = dataload.train_data(data)

    ## load model
    if model_name.lower() == "resnet50":
        model = ResNet50(num_classes=len(labels)).to(device)
    elif model_name.lower() == "resnet101":
        model_name = ResNet101(num_classes=len(labels)).to(device)
    elif model_name.lower() == 'alexnet':
        model = AlexNet(num_classes=len(labels)).to(device)


    ## optimizer & loss function
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)
    elif optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.1)
    loss_fn = nn.CrossEntropyLoss()

    ## training
    training.train(model=model, train_loader=train_loader, optimizer=optimizer, loss_fn=loss_fn, num_epoch=epochs, device=device)

    ## save model
    if model_name.lower() == "resnet50":
        torch.save(model.state_dict(), save_path + f'/{model_name}_{img_size}_ep{epochs}.pt')
    elif model_name.lower() == "resnet101":
        torch.save(model.state_dict(), save_path + f'/{model_name}_{img_size}_ep{epochs}.pt')
    elif model_name.lower() == 'alexnet':
        torch.save(model.state_dict(), save_path + f'/{model_name}_{img_size}_ep{epochs}.pt')

    # ## test
    # # test_loader = dataload.test_data(data)
    # model.load_state_dict(torch.load('weight/resnet101_house_128_100.pt'))
    # training.val(model=model, test_loader=test_loader, device=device)





