"""
This is a boilerplate pipeline 'augmenter_dataset'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # make sure to add the nodes in the right order

        node(
            func = augment_dataset,
            inputs=["params:data_path", "params:augmented_train_path"],
            outputs="enh_data",
            name="augment_dataset"
        )
    ])
