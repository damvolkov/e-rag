"""Indexing pipeline: parse, chunk hierarchically, embed, and store."""

from docling_haystack.converter import DoclingConverter, ExportType
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

from e_rag.core.settings import settings as st


def run_indexing(
    sources: list[str],
    parent_store: PgvectorDocumentStore,
    child_store: PgvectorDocumentStore,
) -> dict[str, int]:
    """Parse documents, create parent-child hierarchy, embed children, store both."""
    converter = DoclingConverter(export_type=ExportType.MARKDOWN)
    parent_splitter = DocumentSplitter(
        split_by="word",
        split_length=st.PARENT_SPLIT_LENGTH,
        split_overlap=st.PARENT_SPLIT_OVERLAP,
    )
    child_splitter = DocumentSplitter(
        split_by="word",
        split_length=st.CHILD_SPLIT_LENGTH,
        split_overlap=st.CHILD_SPLIT_OVERLAP,
    )
    embedder = SentenceTransformersDocumentEmbedder(model=st.EMBEDDING_MODEL)
    embedder.warm_up()

    parsed = converter.run(paths=sources)
    documents: list[Document] = parsed["documents"]

    parent_chunks = parent_splitter.run(documents=documents)["documents"]
    for parent in parent_chunks:
        parent.meta["doc_level"] = "parent"

    DocumentWriter(document_store=parent_store).run(documents=parent_chunks)

    all_children: list[Document] = []
    for parent in parent_chunks:
        children = child_splitter.run(documents=[parent])["documents"]
        for child in children:
            child.meta["parent_id"] = parent.id
            child.meta["doc_level"] = "child"
        all_children.extend(children)

    embedded = embedder.run(documents=all_children)
    DocumentWriter(document_store=child_store).run(documents=embedded["documents"])

    return {"parents": len(parent_chunks), "children": len(all_children)}
