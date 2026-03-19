"""Document store factory for pgvector."""

from haystack.utils import Secret
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

from e_rag.core.settings import settings as st


def create_parent_store(*, recreate: bool = False) -> PgvectorDocumentStore:
    """Create pgvector store for parent chunks."""
    return PgvectorDocumentStore(
        connection_string=Secret.from_token(st.PG_CONN_STR),
        table_name=st.PARENT_TABLE,
        embedding_dimension=st.EMBEDDING_DIMENSION,
        vector_function="cosine_similarity",
        search_strategy="hnsw",
        recreate_table=recreate,
        hnsw_index_name="parent_hnsw_index",
        keyword_index_name="parent_keyword_index",
    )


def create_child_store(*, recreate: bool = False) -> PgvectorDocumentStore:
    """Create pgvector store for child chunks."""
    return PgvectorDocumentStore(
        connection_string=Secret.from_token(st.PG_CONN_STR),
        table_name=st.CHILD_TABLE,
        embedding_dimension=st.EMBEDDING_DIMENSION,
        vector_function="cosine_similarity",
        search_strategy="hnsw",
        recreate_table=recreate,
        hnsw_index_name="child_hnsw_index",
        keyword_index_name="child_keyword_index",
    )
