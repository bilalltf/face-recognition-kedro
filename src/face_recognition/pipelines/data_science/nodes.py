"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.4
"""

from typing import Dict, Tuple
import numpy as np
from kedro.extras.datasets.text import TextDataSet
from kedro.extras.datasets.pickle import PickleDataSet
from PIL import Image
import pandas as pd
from torchvision import transforms, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from kedro.io import AbstractDataSet


from custom_text_data_set import CustomTextDataSet
from .face_recognition import preprocessing, FaceFeaturesExtractor, FaceRecogniser
import tqdm


def dataset_to_embeddings(dataset, features_extractor):
    transform = transforms.Compose([
        preprocessing.ExifOrientationNormalize(),
        transforms.Resize(1024)
    ])

    embeddings = []
    labels = []
    for img_path, label in tqdm.tqdm(dataset.samples):
        # print(img_path)
        _, embedding = features_extractor(transform(Image.open(img_path).convert('RGB')))
        if embedding is None:
            continue
        if embedding.shape[0] > 1:
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


def prepare_embeddings(train_path: str, augmented_train_path: str, augment: bool, data_sets: Dict[str, AbstractDataSet]) -> Tuple:
    features_extractor = FaceFeaturesExtractor()

    # use augmented dataset if augment is True
    data_path = augmented_train_path if augment else train_path
    dataset = datasets.ImageFolder(data_path)

    # extract embeddings
    embeddings, labels = dataset_to_embeddings(dataset, features_extractor)

    # normalise labels
    dataset.class_to_idx = normalise_dict_keys(dataset.class_to_idx)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    labels = list(map(lambda idx: idx_to_class[idx], labels))

    # get the labels_data_set instance from the data_sets
    labels_data_set = data_sets["labels"]
    # save labels to the labels_data_set instance
    labels_data_set.save(np.array(labels, dtype=np.str).reshape(-1, 1))
    return embeddings, labels_data_set, dataset.class_to_idx








def train(embeddings:TextDataSet, labels:TextDataSet, class_to_idx:PickleDataSet, grid_search:bool):
    features_extractor = FaceFeaturesExtractor()
    
    softmax = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=10, max_iter=10000, verbose=1)
    if grid_search:
        clf = GridSearchCV(
            estimator=softmax,
            param_grid={'C': [0.1, 1, 10, 100, 1000]},
            cv=3
        )
    else:
        clf = softmax
    clf.fit(embeddings, labels)


    labels = labels.tolist()
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    target_names = map(lambda i: i[1], sorted(idx_to_class.items(), key=lambda i: i[0]))
    print("train accuracy: {}".format(metrics.accuracy_score(labels, clf.predict(embeddings))))
    print("train precision: {}".format(metrics.precision_score(labels, clf.predict(embeddings), average='weighted')))
    print("train recall: {}".format(metrics.recall_score(labels, clf.predict(embeddings), average='weighted')))
    print("train f1: {}".format(metrics.f1_score(labels, clf.predict(embeddings), average='weighted')))

    return FaceRecogniser(features_extractor, clf, idx_to_class)

def evaluate(model:FaceRecogniser, test_embeddings:TextDataSet, test_labels:TextDataSet, test_class_to_idx:PickleDataSet, grid_search:bool, augment:bool):


    labels = test_labels.tolist()
    idx_to_class = {v: k for k, v in test_class_to_idx.items()}
    target_names = map(lambda i: i[1], sorted(idx_to_class.items(), key=lambda i: i[0]))

    # save the metrics to a csv file
    gs=1 if grid_search else 0
    ag=1 if augment else 0

    metrics = {
        'accuracy': metrics.accuracy_score(labels, model.predict(test_embeddings)),
        'precision': metrics.precision_score(labels, model.predict(test_embeddings), average='weighted'),
        'recall': metrics.recall_score(labels, model.predict(test_embeddings), average='weighted'),
        'f1_score': metrics.f1_score(labels, model.predict(test_embeddings), average='weighted'),
        'grid_search': gs,
        'augmentation': ag
    }
    for key, value in metrics.items():
        print("{}: {}".format(key, value))
    # convert metrics to dataframe
    metrics_df = pd.DataFrame(metrics, index=[0])
    return metrics_df