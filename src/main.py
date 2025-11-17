import os
from langchain.chat_models import init_chat_model
from vectorstore import VectorStore
from embedding import EmbeddingManager
from rag import RAGRetriever
from data_loader import process_all_pdfs,split_documents

vectorstore = VectorStore()
embedding_manager = EmbeddingManager()
rag_retrieved = RAGRetriever(vectorstore, embedding_manager)

all_pdf_documents = process_all_pdfs("../data")
chunks = split_documents(all_pdf_documents)

texts = [doc.page_content for doc in chunks]

embeddings = embedding_manager.generate_embeddings(texts)

vectorstore.ajouter_documents(chunks,embeddings)

model = init_chat_model(
    model="google/gemma-3n-e4b",
    model_provider="openai",
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
    temperature=0.1,
    max_tokens=1024
)


def rag_complete(query,retriever,llm,top_k=3):
    results = retriever.rechercher(query,top_k=top_k)
    context = "\n\n".join([doc['contenu'] for doc in results]) if results else ""
    if not context:
        return "Aucun contexte pertinent pour répondre à la question"

    prompt=f"Use the following context to answer the question concisely. Context: {context} Question:{query} Answer:"

    response = llm.invoke([prompt.format(context=context,query=query)])

    return response.content


answer = rag_complete("Qui es le Petit Prince ?",rag_retrieved,model)
print("Réponse: ",answer)
