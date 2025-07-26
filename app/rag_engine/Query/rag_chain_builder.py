from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

def build_rag_chain(vectordb, llm, embeddings_model=None):
    """
    Build RAG chain with provided vectordb, llm, and optional embeddings model
    """
    def format_context(docs_result):
        if not docs_result or 'documents' not in docs_result or not docs_result['documents']:
            return "No relevant documents found."
        
        documents = docs_result['documents'][0]  # First query result
        metadatas = docs_result['metadatas'][0]  # First query metadata
        
        formatted_docs = []
        for doc, meta in zip(documents, metadatas):
            source = meta.get('source', 'Unknown')
            formatted_docs.append(f"{doc}\n[Source: {source}]")
        
        return "\n\n".join(formatted_docs)

    def retrieve_context(input_dict):
        query = input_dict["question"]
        
        # Use provided embeddings model or create new one (fallback)
        if embeddings_model:
            query_embedding = embeddings_model.embed_query(query)
        else:
            # Fallback: only if embeddings_model not provided
            from langchain_huggingface import HuggingFaceEmbeddings
            temp_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            query_embedding = temp_embeddings.embed_query(query)
        
        # Query the vectorstore with actual embedding
        docs_result = vectordb.get_doc_collection().query(query_embeddings=[query_embedding],n_results=4)

        
        return {
            "context": format_context(docs_result),
            "question": query
        }
    

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Use the following context to answer the question. If the context doesn't contain relevant information, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
    )

    # Build the chain
    chain = retrieve_context | prompt | llm
    
    return chain