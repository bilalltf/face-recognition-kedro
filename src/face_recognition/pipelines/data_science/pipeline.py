"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import prepare_embeddings, train
def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
            func=prepare_embeddings,
            inputs=["params:data_path"],
            outputs=["embeddings", "labels", "class_to_idx"],
            name="prepare_embeddings"
        ),

        node(
            func=train,
            inputs=["embeddings", "labels", "class_to_idx", "params:grid_search"],
            outputs="model",
            name="train_model"
        )


    ])
