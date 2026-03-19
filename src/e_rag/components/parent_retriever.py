"""Custom Haystack component for parent document retrieval."""

from haystack import Document, component
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore


@component
class ParentDocumentRetriever:
    """Resolve child chunks back to their parent documents."""

    def __init__(self, document_store: PgvectorDocumentStore) -> None:
        self.document_store = document_store

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """Look up parent documents by parent_id from child metadata."""
        parent_ids = list(frozenset(
            doc.meta["parent_id"]
            for doc in documents
            if "parent_id" in doc.meta
        ))

        if not parent_ids:
            return {"documents": []}

        parents = self.document_store.filter_documents(
            filters={
                "operator": "OR",
                "conditions": [
                    {"field": "id", "operator": "==", "value": pid}
                    for pid in parent_ids
                ],
            }
        )

        return {"documents": parents}
