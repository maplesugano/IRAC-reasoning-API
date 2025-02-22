from pydantic_settings import BaseSettings
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (
    OpenAIChatPromptExecutionSettings
)
import yaml
from typing import Optional

class Settings(BaseSettings):
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_ORG_ID: Optional[str] = None
    SERVICE_ID: str  = None

    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: Optional[str] = None
    PINECONE_ENV: Optional[str] = None   

    ORS_API_KEY: Optional[str] = None

    GRAPHRAG_LLM_MODEL: str = "gpt-4o-mini"
    GRAPHRAG_LLM_API_BASE: Optional[str] = None
    GRAPHRAG_EMBEDDING_MODEL: str = "text-embedding-3-small"
    GRAPHRAG_EMBEDDING_API_BASE: Optional[str] = None

    GRAPHRAG_CLAIM_EXTRACTION_ENABLED: bool = False
    INPUT_DIR: str = "./artifacts"
    COMMUNITY_LEVEL: int = 2
    RESPONSE_TYPE: str = "single paragraph"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

def setup(settings: Settings):
    """
    Initialize the ChatOpenAI and OpenAIEmbedding instances based on the settings.

    Args:
        settings: The settings object containing configuration details.

    Returns:
        A tuple of (llm, text_embedder).
    """
    common_params = {
        "max_retries": 20,
    }
        
    llm = ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        api_base=settings.GRAPHRAG_LLM_API_BASE,
        model=settings.GRAPHRAG_LLM_MODEL,
        api_type=OpenaiApiType.OpenAI,
        **common_params,
    )

    text_embedder = OpenAIEmbedding(
        api_key=settings.OPENAI_API_KEY,
        api_base=settings.GRAPHRAG_EMBEDDING_API_BASE,
        api_type=OpenaiApiType.OpenAI,
        model=settings.GRAPHRAG_EMBEDDING_MODEL,
        deployment_name=settings.GRAPHRAG_EMBEDDING_MODEL,
        **common_params,
    )
    
    chat_completion = OpenAIChatCompletion(
        ai_model_id=settings.GRAPHRAG_LLM_MODEL,
        service_id=settings.SERVICE_ID,
        api_key=settings.OPENAI_API_KEY,
        org_id=settings.OPENAI_ORG_ID
    )

    execution_settings = OpenAIChatPromptExecutionSettings()
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    return llm, text_embedder, chat_completion, execution_settings


def load_settings_from_yaml(yaml_file: str) -> Settings:
    """
    Load settings from a YAML file and override with environment variables.

    Args:
        yaml_file (str): Path to the YAML configuration file.        

    Returns:
        Settings: Pydantic Settings object with merged configurations.
    """
    with open(yaml_file, 'r', encoding='utf-8') as file:
        config_dict = yaml.safe_load(file)
    return Settings(**config_dict)
