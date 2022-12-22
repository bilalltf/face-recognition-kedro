"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.4
"""
import os
import joblib
import numpy as np
from kedro.extras.datasets.text import TextDataSet
from kedro.extras.datasets.pickle import PickleDataSet
from PIL import Image
from torchvision import transforms, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from .face_recognition import preprocessing, FaceFeaturesExtractor, FaceRecogniser


def dataset_to_embeddings(dataset, features_extractor):
    transform = transforms.Compose([
        preprocessing.ExifOrientationNormalize(),
        transforms.Resize(1024)
    ])

    embeddings = []
    labels = []
    for img_path, label in dataset.samples:
        print(img_path)
        _, embedding = features_extractor(transform(Image.open(img_path).convert('RGB')))
        if embedding is None:
            print("Could not find face on {}".format(img_path))
            continue
        if embedding.shape[0] > 1:
            print("Multiple faces detected for {}, taking one with highest probability".format(img_path))
            embedding = embedding[0, :]
        embeddings.append(embedding.flatten())
        labels.append(label)

    return np.stack(embeddings), labels

def normalise_string(string):
    return string.lower().replace(' ', '_')

def normalise_dict_keys(dictionary):
    new_dict = dict()
    for key in dictionary.keys():
        new_dict[normalise_string(key)] = dictionary[key]
    return new_dict

def prepare_embeddings(data_path:str):
    features_extractor = FaceFeaturesExtractor()
    dataset = datasets.ImageFolder(data_path)
    embeddings, labels = dataset_to_embeddings(dataset, features_extractor)
    dataset.class_to_idx = normalise_dict_keys(dataset.class_to_idx)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    labels = list(map(lambda idx: idx_to_class[idx], labels))
    return embeddings, np.array(labels, dtype=np.str).reshape(-1, 1), dataset.class_to_idx

def train(embeddings:TextDataSet, labels:TextDataSet, class_to_idx:PickleDataSet, grid_search:bool):
    features_extractor = FaceFeaturesExtractor()
    softmax = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=10, max_iter=10000)
    if grid_search:
        clf = GridSearchCV(
            estimator=softmax,
            param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
            cv=3
        )
    else:
        clf = softmax
    clf.fit(embeddings, labels)

    labels = labels.tolist()
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    target_names = map(lambda i: i[1], sorted(idx_to_class.items(), key=lambda i: i[0]))
    print(metrics.classification_report(labels, clf.predict(embeddings), target_names=list(target_names)))


    return FaceRecogniser(features_extractor, clf, idx_to_class)
