"""
This is a boilerplate pipeline 'augmenter_dataset'
generated using Kedro 0.18.3
"""
import albumentations as A
import cv2
import random
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil
from PIL import Image

# split the dataset into train and test, the test set is 20% of the dataset, the total images are in the path data_path, copy 20% of them in the test folder with the path test_path and the rest in the train folder with the path train_path
# copy the images directly using shutil.copy
# first delete the images in test and train folders if they exist

def split_dataset(data_path, train_path, test_path):
    print("Splitting dataset...")
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    os.mkdir(train_path)
    os.mkdir(test_path)
    subfolders = [f.path for f in os.scandir(data_path) if f.is_dir()]
    test_images = subfolders[:int(len(subfolders)*0.5)]
    train_images = subfolders[int(len(subfolders)*0.5):]
    for f, k in zip(test_images, tqdm(range(len(test_images)))):
        shutil.copytree(f, test_path + "/" +  os.path.split(f)[-1])
        
    for f, k in zip(train_images, tqdm(range(len(train_images)))):
        shutil.copytree(f, train_path + "/" +  os.path.split(f)[-1])
    return True


def augment_dataset(train_path, augmented_train_path, split_dataset_done):
    if os.path.exists(augmented_train_path):
        shutil.rmtree(augmented_train_path)
    os.mkdir(augmented_train_path)
    data = []
    print("Augmenting train dataset...")

    subfolders = [f.path for f in os.scandir(train_path) if f.is_dir()]
    for f,j in zip(subfolders, tqdm(range(len(subfolders)))):
        person = path_to_name_no_extension(f)
        images = [i.path for i in os.scandir(f) if i.is_file()]
        
        for img in images:
            save_path = augmented_train_path + "/" + person

            image = cv2.imread(img)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            name = save_path+".jpg"
            cv2.imwrite(name, image)
            data.append([name, person])

            random.seed(7)

            #affine
            affine_trf = A.Affine(always_apply=False, p=1.0, cval=128)
            affined_img = affine_trf(image=image)['image']
            name = save_path +"_affine"+".jpg"
            cv2.imwrite(name, affined_img)
            data.append([name, person])

            # contrast+brightness
            br_ctr = A.ColorJitter(always_apply=False, p=1.0, contrast=0.5, brightness=0.5)
            br_ctr_img = br_ctr(image=image)['image']
            
            # pick a random noise type
            noise_types = ["gauss", "s&p", "poisson", "speckle"]
            noise_type = random.choice(noise_types)
            br_ctr_img = noisy(noise_type, br_ctr_img)
            transformed_image = Image.fromarray((br_ctr_img * 255).astype(np.uint8))
            name = save_path +"_contrastBrightness"+".jpg"
            # cv2.imwrite(name, br_ctr_img)
            transformed_image.save(name, dpi=(600,600), quality=100, subsampling=0)

            data.append([name, person])

    data = np.array(data)
    data = pd.DataFrame(data, columns = ['file','person'])
    return data

def path_to_name_no_extension(path):#if folder, return the name of the folder
    return os.path.split(path)[-1].split('.')[0]

def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
          # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

          # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2**np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
        return noisy