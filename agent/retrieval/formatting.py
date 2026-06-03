from typing import List

from models import RetrievedDocument


def format_documents_as_context(
    documents: List[RetrievedDocument],
    empty_message: str = "No context documents available.",
) -> str:
    if not documents:
        return empty_message

    context_lines = []
    for i, doc in enumerate(documents, start=1):
        context_lines.append(
            f"[Document {i}] {doc.filename} (relevance: {doc.similarity_score:.2%})\n"
            f"Content: {doc.text}\n"
        )

    return "\n".join(context_lines)
