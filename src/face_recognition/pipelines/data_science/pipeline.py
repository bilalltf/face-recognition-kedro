"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
import numpy as np
from .nodes import prepare_embeddings, train, evaluate
from kedro.io import AbstractDataSet, DataSetError
from typing import Any


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
            func=prepare_embeddings,
            inputs=["params:train_path", "params:augmented_train_path", "params:augment", "data_sets"],
            outputs=["embeddings", "labels", "class_to_idx"],
            name="prepare_embeddings",
        ),

        node(
            func=train,
            inputs=["embeddings", "labels", "class_to_idx", "params:grid_search"],
            outputs="model",
            name="train_model"
        ),
       node(
            func=prepare_embeddings,
            inputs=["params:test_path", "params:augmented_test_path", "params:augment", "data_sets"],
            outputs=["test_embeddings", "test_labels", "test_class_to_idx"],
            name="prepare_test_embeddings",
        ),
        node(
            func=evaluate,
            inputs=["model", "test_embeddings", "test_labels", "test_class_to_idx", "params:grid_search", "params:augment"],
            outputs="eval",
            name="evaluate_model"

        )


    ])
