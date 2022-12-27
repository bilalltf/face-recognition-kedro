"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline
from .pipelines import data_science as ds, data_engineering as de



def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.
    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # data_engineering = de.create_pipeline()
    data_science = ds.create_pipeline()
    data_engineering = de.create_pipeline()
    
    return {"de": data_engineering,"ds": data_science, "__default__":data_engineering}
