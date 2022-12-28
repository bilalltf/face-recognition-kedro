"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
            func=prepare_embeddings,
            inputs=["params:data_path", "params:augmented_train_path", "params:augment"],
            outputs=["embeddings", "labels", "class_to_idx"],
            name="prepare_embeddings",
        ),

        node(
            func=train,
            inputs=["embeddings", "labels", "class_to_idx", "params:grid_search", "params:model_path"],
            outputs="model",
            name="train_model"
        ),
        node(
            func=eval,
            inputs=["model", "embeddings", "labels", "class_to_idx", "params:grid_search", "params:augment"],
            outputs="eval",
            name="evaluate_model"

        )


    ])
