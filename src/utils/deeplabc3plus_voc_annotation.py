import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm


def init_format(label_path):
    save_path = os.path.join(label_path).rsplit('/',1)[0]
    trainval_percent    = 1
    train_percent       = 0.9
    random.seed(0)
    print("Generate txt in ImageSets.")
    segfilepath     = os.path.join(label_path)
    saveBasePath    = os.path.join(save_path, 'Segmentation')
    os.makedirs(saveBasePath, exist_ok=True)
    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num     = len(total_seg)  
    list    = range(num)  
    tv      = int(num*trainval_percent)  
    tr      = int(tv*train_percent)  
    trainval= random.sample(list,tv)  
    train   = random.sample(trainval,tr)  
    
    print("train and val size",tv)
    print("traub suze",tr)
    ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
    ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    
    for i in list:  
        name = total_seg[i][:-4]+'\n'  
        if i in trainval:  
            ftrainval.write(name)  
            if i in train:  
                ftrain.write(name)  
            else:  
                fval.write(name)  
        else:  
            ftest.write(name)  
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()
    print("Generate txt in ImageSets done.")

    print("Check datasets format, this may take a while.")

    classes_nums        = np.zeros([256], np.int_)
    for i in tqdm(list):
        name            = total_seg[i]
        png_file_name   = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError("There are no endswith .png file")
        
        png             = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("label image%sshape is %s, It's no grayscale"%(name, str(np.shape(png))))


        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)
            
    print("pix value and num:")
    print('-' * 37)
    print("| %15s | %15s |"%("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |"%(str(i), str(classes_nums[i])))
            print('-' * 37)
    
    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("pix value error, need [bg value==0, target value==1,2,3...]")

    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("There are no target pix")

    print("JPEGImages should have .jpg file, SegmentationClass should have .png file.")
