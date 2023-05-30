from src.config import Config
from src.Classification import *
from src.Objectdetection import *
from src.Segmentation import *


# int = task, epochs, batch_size
# float = lr
# string = model, dir_data, dir_label

if __name__ =="__main__":

    cfg = Config('./training_parameter.txt')
    config = cfg.hyperparameter()
    print(config)

    if config['task'] == 0: ## classification
        weight = train_classification(config)

    if config['task'] == 1: ## object detection
        label = ["building","vinyl house"]
        weight = train_objectdetection(config, label)

    if config['task'] == 2: ## segmentation
        label_num = 2
        weight = train_segmentation(config, label_num)