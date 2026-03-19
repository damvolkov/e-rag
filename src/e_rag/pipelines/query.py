"""Query pipeline: hybrid retrieval with parent document resolution."""

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack_integrations.components.retrievers.pgvector import (
    PgvectorEmbeddingRetriever,
    PgvectorKeywordRetriever,
)
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

from e_rag.components.parent_retriever import ParentDocumentRetriever
from e_rag.core.settings import settings as st


def build_query_pipeline(
    child_store: PgvectorDocumentStore,
    parent_store: PgvectorDocumentStore,
) -> Pipeline:
    """Build hybrid retrieval pipeline with parent document resolution."""
    pipeline = Pipeline()

    pipeline.add_component(
        "text_embedder",
        SentenceTransformersTextEmbedder(model=st.EMBEDDING_MODEL),
    )
    pipeline.add_component(
        "embedding_retriever",
        PgvectorEmbeddingRetriever(document_store=child_store, top_k=st.TOP_K),
    )
    pipeline.add_component(
        "keyword_retriever",
        PgvectorKeywordRetriever(document_store=child_store, top_k=st.TOP_K),
    )
    pipeline.add_component(
        "joiner",
        DocumentJoiner(
            join_mode="reciprocal_rank_fusion",
            weights=[st.KEYWORD_WEIGHT, st.SEMANTIC_WEIGHT],
        ),
    )
    pipeline.add_component(
        "parent_retriever",
        ParentDocumentRetriever(document_store=parent_store),
    )

    pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
    pipeline.connect("keyword_retriever.documents", "joiner.documents")
    pipeline.connect("embedding_retriever.documents", "joiner.documents")
    pipeline.connect("joiner.documents", "parent_retriever.documents")

    return pipeline
