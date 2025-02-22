import json
from typing import Union, List, Dict, Any
import pandas as pd
from graphrag.query.structured_search.base import SearchResult
from utils.constants import (
    COMMUNITY_REPORT_TABLE,
    ENTITY_TABLE,
    ENTITY_EMBEDDING_TABLE,
    RELATIONSHIP_TABLE,
    COVARIATE_TABLE,
    TEXT_UNIT_TABLE,
    COMMUNITY_TABLE
)

def load_parquet_files(input_dir: str, claim_extraction_enabled: bool):
    """
    Loads multiple parquet files from the specified input directory.
    Parameters:
        input_dir (str): The directory where the parquet files are located.
        claim_extraction_enabled (bool): Flag to determine if the covariate table should be loaded.
    Returns:
        tuple: A tuple containing the following DataFrames:
            - entity_df: DataFrame containing entity data.
            - entity_embedding_df: DataFrame containing entity embedding data.
            - report_df: DataFrame containing community report data.
            - relationship_df: DataFrame containing relationship data.
            - covariate_df: DataFrame containing covariate data (empty if claim_extraction_enabled is False).
            - text_unit_df: DataFrame containing text unit data.
            - community_df: DataFrame containing community data.
    """
    entity_df = pd.read_parquet(f"{input_dir}/{ENTITY_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{input_dir}/{ENTITY_EMBEDDING_TABLE}.parquet")
    report_df = pd.read_parquet(f"{input_dir}/{COMMUNITY_REPORT_TABLE}.parquet")
    relationship_df = pd.read_parquet(f"{input_dir}/{RELATIONSHIP_TABLE}.parquet")    
    covariate_df = pd.read_parquet(f"{input_dir}/{COVARIATE_TABLE}.parquet") if claim_extraction_enabled else pd.DataFrame()
    text_unit_df = pd.read_parquet(f"{input_dir}/{TEXT_UNIT_TABLE}.parquet")
    community_df = pd.read_parquet(f"{input_dir}/{COMMUNITY_TABLE}.parquet")

    return entity_df, entity_embedding_df, report_df, relationship_df, covariate_df, text_unit_df, community_df

def convert_response_to_string(response: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> str:
    """
    Convert a response that can be a string, dictionary, or list of dictionaries to a string.
    """
    if isinstance(response, (dict, list)):
        return json.dumps(response)
    elif isinstance(response, str):
        return response
    else:
        return str(response)

def recursively_convert(obj: Any) -> Any:
    """
    Recursively converts a given object into a dictionary or list format.
    If the input object is a pandas DataFrame, it converts it to a list of dictionaries.
    If the input object is a list, it recursively converts each element in the list.
    If the input object is a dictionary, it recursively converts each value in the dictionary.
    Otherwise, it returns the object as is.
    Parameters:
        obj (Any): The input object to be converted. It can be a pandas DataFrame, list, dictionary, or any other type.
    Returns:
        Any: The converted object in dictionary or list format, or the original object if it is not a DataFrame, list, or dictionary.
    """
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, list):
        return [recursively_convert(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: recursively_convert(value) for key, value in obj.items()}
    return obj

def process_context_data(context_data: Union[str, List[pd.DataFrame], Dict, pd.DataFrame]) -> Any:
    """
    Process context data to convert it to a format that can be serialized.
    Parameters:
        context_data (Union[str, List[pd.DataFrame], Dict, pd.DataFrame]): The context data to be processed.
    Returns:
        Any: The processed context data in a format that can be serialized.
    """
    if isinstance(context_data, str):
        return context_data
    if isinstance(context_data, pd.DataFrame):
        return context_data.to_dict(orient="records")
    if isinstance(context_data, (list, dict)):
        return recursively_convert(context_data)
    return None

def serialize_search_result(search_result: SearchResult) -> Dict[str, Any]:
    """
    Serialise a search result object into a dictionary.
    Parameters:
        search_result (SearchResult): The search result object to be serialised.
    Returns:
        Dict[str, Any]: The serialised search result object
    """
    return {
        "response": search_result.response,
        "context_data": process_context_data(search_result.context_data),
        "context_text": search_result.context_text,
        "completion_time": search_result.completion_time,
        "llm_calls": search_result.llm_calls,
        "prompt_tokens": search_result.prompt_tokens
    }