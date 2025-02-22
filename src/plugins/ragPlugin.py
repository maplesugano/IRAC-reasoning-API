from semantic_kernel.functions import kernel_function
from utils.parquet_utils import load_parquet_files, convert_response_to_string, process_context_data

from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
    read_indexer_communities
)

import tiktoken

class RAGPlugin:

    raw_RAG_result = None

    def __init__(self, settings, text_embedder, llm):
        self.token_encoder = tiktoken.get_encoding("cl100k_base")

        self.CLAIM_EXTRACTION_ENABLED = settings.GRAPHRAG_CLAIM_EXTRACTION_ENABLED
        self.RESPONSE_TYPE = settings.RESPONSE_TYPE

        entity_df, entity_embedding_df, report_df, relationship_df, covariate_df, text_unit_df, community_df = load_parquet_files(settings.INPUT_DIR, self.CLAIM_EXTRACTION_ENABLED)
        self.entities = read_indexer_entities(entity_df, entity_embedding_df, settings.COMMUNITY_LEVEL)
        self.relationships = read_indexer_relationships(relationship_df)
        self.claims = read_indexer_covariates(covariate_df) if self.CLAIM_EXTRACTION_ENABLED else []
        self.reports = read_indexer_reports(report_df, entity_df, settings.COMMUNITY_LEVEL)
        self.text_units = read_indexer_text_units(text_unit_df)
        self.communities = read_indexer_communities(community_df, entity_df, report_df)

        self.description_embedding_store = LanceDBVectorStore(
            collection_name="default-entity-description",
        )
        self.description_embedding_store.connect(db_uri=f"{settings.INPUT_DIR}/lancedb")

        self.full_content_embedding_store = LanceDBVectorStore(
            collection_name="default-community-full_content",
        )
        self.full_content_embedding_store.connect(db_uri=f"{settings.INPUT_DIR}/lancedb")

        self.text_embedder = text_embedder
        self.llm = llm

        context_builder = LocalSearchMixedContext(
            community_reports=self.reports,
            text_units=self.text_units,
            entities=self.entities,
            relationships=self.relationships,
            covariates={"claims": self.claims} if self.CLAIM_EXTRACTION_ENABLED else None,
            entity_text_embeddings=self.description_embedding_store,
            embedding_vectorstore_key=EntityVectorStoreKey.ID,
            text_embedder=self.text_embedder,
            token_encoder=self.token_encoder,
        )
    
        self.local_search_engine = LocalSearch(
            llm=self.llm,
            context_builder=context_builder,
            token_encoder=self.token_encoder,
            llm_params={
                "max_tokens": 2_000,
                "temperature": 0.0,
            },
            context_builder_params={
                "text_unit_prop": 0.5,
                "community_prop": 0.1,
                "conversation_history_max_turns": 5,
                "conversation_history_user_turns_only": True,
                "top_k_mapped_entities": 10,
                "top_k_relationships": 10,
                "include_entity_rank": True,
                "include_relationship_weight": True,
                "include_community_rank": False,
                "return_candidate_context": False,
                "embedding_vectorstore_key": EntityVectorStoreKey.ID,
                "max_tokens": 12_000,
            },
            response_type=self.RESPONSE_TYPE,
        )    
    
    @kernel_function
    async def local_search(self, query: str):
        """
        Performs Search about the geographical / temporal transportation and intrastructure information.

        Args:
            query (str): The query to search for.
        
        Returns:
            dict: The response dictionary
        """
        print(f"Calling local search with query: {query}")
        try:
            result = await self.local_search_engine.asearch(query)        
            response_dict = {
                "response": convert_response_to_string(result.response),
                "context_data": process_context_data(result.context_data),
                "context_text": result.context_text,
                "completion_time": result.completion_time,
                "llm_calls": result.llm_calls,
                "llm_calls_categories": result.llm_calls_categories,
                "output_tokens":result.output_tokens,
                "output_tokens_categories":result.output_tokens_categories,
                "prompt_tokens": result.prompt_tokens,
                "prompt_tokens_categories": result.prompt_tokens_categories
            }
            self.raw_RAG_result = response_dict
            return response_dict
        except Exception as e:
            raise Exception(f"Error performing local search.")
