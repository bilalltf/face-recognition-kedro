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

# def split_dataset(data_path, train_path, test_path):
#     print("Splitting dataset...")
#     if os.path.exists(train_path):
#         shutil.rmtree(train_path)
#     if os.path.exists(test_path):
#         shutil.rmtree(test_path)
#     os.mkdir(train_path)
#     os.mkdir(test_path)
#     subfolders = [f.path for f in os.scandir(data_path) if f.is_dir()]
#     test_images = subfolders[:int(len(subfolders)*0.5)]
#     train_images = subfolders[int(len(subfolders)*0.5):]
#     for f, k in zip(test_images, tqdm(range(len(test_images)))):
#         shutil.copytree(f, test_path + "/" +  os.path.split(f)[-1])
        
#     for f, k in zip(train_images, tqdm(range(len(train_images)))):
#         shutil.copytree(f, train_path + "/" +  os.path.split(f)[-1])
#     return True

def generate_test_set(train_path, test_path):
    data = []
    print("generating test dataset...")
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    os.mkdir(test_path)

    subfolders = [f.path for f in os.scandir(train_path) if f.is_dir()]
    for f,j in zip(subfolders, tqdm(range(len(subfolders)))):
        person = path_to_name_no_extension(f)
        images = [i.path for i in os.scandir(f) if i.is_file()]
        
        for img in images:
            # create directory for the person
            if not os.path.exists(test_path + "/" + person):
                os.mkdir(test_path + "/" + person)
            save_path = test_path + "/" + person + "/" + person

            image = cv2.imread(img)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Random affine
            affine_trf =  A.ShiftScaleRotate(always_apply=True, p=1.0, shift_limit=0.09, scale_limit=0.5, rotate_limit=(-45, -35), border_mode=0, value=(255, 255, 255), mask_value=None)
            contrast = A.RandomContrast(limit=(0.5, 0.7), p=1.0)
            affined_img = affine_trf(image=image)['image']
            cont_image = contrast(image=affined_img)['image']
            transformed_image = Image.fromarray((cont_image).astype(np.uint8))
            name = save_path + ".jpg"
            transformed_image.save(name, dpi=(300,300), quality=100, subsampling=0)
            data.append([name, person])



def augment_dataset(train_path, augmented_train_path):
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
            # create directory for the person
            if not os.path.exists(augmented_train_path + "/" + person):
                os.mkdir(augmented_train_path + "/" + person)
            save_path = augmented_train_path + "/" + person + "/" + person

            image = cv2.imread(img)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # copy original image to the new folder using shutil
            name = save_path +".jpg"
            shutil.copy(img, name)
            data.append([name, person])
            random.seed(7)

            # Random affine
            affine_trf =  A.ShiftScaleRotate(always_apply=True, p=1.0, shift_limit=0.09, scale_limit=1, rotate_limit=(35, 45), border_mode=0, value=(0, 0, 0), mask_value=None)
            affined_img = affine_trf(image=image)['image']
            transformed_image = Image.fromarray((affined_img).astype(np.uint8))
            name = save_path +"_affine"+".jpg"
            transformed_image.save(name, dpi=(300,300), quality=100, subsampling=0)
            data.append([name, person])

            # ColorJitter and noise
            br_ctr = A.ColorJitter(always_apply=False, p=1.0, contrast=0.5, brightness=0.5, hue=0.5, saturation=0.5)
            br_ctr_img = br_ctr(image=image)['image']
            
            # pick a random noise type
            br_ctr_img = noisy("s&p", br_ctr_img)
            transformed_image = Image.fromarray((br_ctr_img).astype(np.uint8))
            name = save_path +"_contrastBrightness_noise"+".jpg"
            transformed_image.save(name, dpi=(300,300), quality=100, subsampling=0)
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