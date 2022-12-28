"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.4
"""

import numpy as np
from kedro.extras.datasets.text import TextDataSet
from kedro.extras.datasets.pickle import PickleDataSet
from PIL import Image
import pandas as pd
from torchvision import transforms, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from .face_recognition import preprocessing, FaceFeaturesExtractor, FaceRecogniser, evaluate
import tqdm


def dataset_to_embeddings(dataset, features_extractor):
    transform = transforms.Compose([
        preprocessing.ExifOrientationNormalize(),
        transforms.Resize(1024)
    ])

    embeddings = []
    labels = []
    for img_path, label in tqdm.tqdm(dataset.samples):
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


def prepare_embeddings(train_path: str, augmented_train_path: str, augment: bool):
    print("generating embeddings...")
    features_extractor = FaceFeaturesExtractor()

    # use augmented dataset if augment is True
    data_path = augmented_train_path if augment else train_path
    dataset = datasets.ImageFolder(data_path)

    # extract embeddings
    embeddings, labels = dataset_to_embeddings(dataset, features_extractor)
    print(embeddings.shape, len(labels))
    # normalise labels
    dataset.class_to_idx = normalise_dict_keys(dataset.class_to_idx)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    labels = list(map(lambda idx: idx_to_class[idx], labels))
    
    return embeddings, np.array(labels, dtype=np.str).reshape(-1, 1), dataset.class_to_idx






def train(embeddings:TextDataSet, labels:TextDataSet, class_to_idx:PickleDataSet, grid_search:bool):
    print("training model...")
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
    if grid_search:
        print("Best parameters set found on training set:")
        print(clf.best_params_)

    labels = labels.tolist()
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    target_names = list(map(lambda i: i[1], sorted(idx_to_class.items(), key=lambda i: i[0])))
    report = metrics.classification_report(labels, clf.predict(embeddings), target_names=target_names, output_dict=True)
    print("Training report:")
    print(f"Accuracy: {report['accuracy']:.3f}")
    print(f"Precision: {report['weighted avg']['precision']:.3f}")
    print(f"Recall: {report['weighted avg']['recall']:.3f}")
    print(f"F1-score: {report['weighted avg']['f1-score']:.3f}")
    print(f"Average confidence: {report['weighted avg']['support']:.3f}")
    return FaceRecogniser(features_extractor, clf, idx_to_class)

def eval(model:FaceRecogniser, embeddings:TextDataSet, labels:TextDataSet, class_to_idx:PickleDataSet, grid_search:bool, augment:bool):
    metrics_dict = {}

    labels = labels.tolist()

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    target_names =list(map(lambda i: i[1], sorted(idx_to_class.items(), key=lambda i: i[0])))
    print(target_names)
    # save the metrics to a csv file
    gs=1 if grid_search else 0
    ag=1 if augment else 0
    # define metrics
    print(labels)
    metrics = evaluate.ModelMetrics(model)
    report = metrics.calculate_metrics(embeddings, labels, target_names)
    print("Test report:")
    print(f"Accuracy: {report['accuracy']:.3f}")
    print(f"Precision: {report['weighted avg']['precision']:.3f}")
    print(f"Recall: {report['weighted avg']['recall']:.3f}")
    print(f"F1-score: {report['weighted avg']['f1-score']:.3f}")
    print(f"Average confidence: {report['weighted avg']['support']:.3f}")

    metrics_dict = {
        "Accuracy": f"{report['accuracy']:.3f}",
        "Precision": f"{report['weighted avg']['precision']:.3f}",
        "Recall": f"{report['weighted avg']['recall']:.3f}",
        "F1-score": f"{report['weighted avg']['f1-score']:.3f}",
        "Average confidence": f"{report['weighted avg']['support']:.3f}%",
        'grid_search': gs,
        'augmentation': ag
    }
    
    
    # convert metrics to dataframe
    metrics_df = pd.DataFrame(metrics_dict, index=[0])
    return metrics_df