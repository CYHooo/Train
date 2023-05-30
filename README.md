# Train
---

## It's for LX project Training Code

### training_parameter.txt:
- Task: 
> classification --> 0, Object detection --> 1, Semantic Segmentation --> 2
    
- model: (model name)
> Classifcation --> resnet50, resnet101, Alexnet 

> Object detection --> yolov8s, Faster R-CNN
    
> Segmentation --> U-net, DeepLabv3+
    
- dir_data & dir_label:
> PATH to images & labels(or masks)

- hyperparmeter:
> lr

> epoch 

> batch_size

> optimizer (adam, sgd)

### training.py
- Training code, get config from '*training_parameter.txt*'

### src
- source code folder for model network and utils files

      |--src    
          |--nets
          |--utils                
