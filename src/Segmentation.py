from src.utils.unet_training import Unet_train
from src.utils.deeplabv3plus_training import Deeplabv3plus_training

def train_segmentation(config,label_num):
    label_num = label_num
    dir_data, dir_label, model, epochs, lr, batch_size, optimizer = config['dir_data'],\
                                                                    config['dir_label'],\
                                                                    config['model'],\
                                                                    config['epochs'],\
                                                                    config['lr'],\
                                                                    config['batch_size'],\
                                                                    config['optimizer']
    
    if model.lower() == 'u-net' or 'unet':
        Unet_train(model, dir_label, label_num, epochs, lr, dir_data, optimizer)
    elif model.lower() == 'deeplabv3+' :
        Deeplabv3plus_training(model, dir_label, label_num, epochs, lr,dir_data, optimizer)