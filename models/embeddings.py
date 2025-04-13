import os
import logging
from typing import List,Union, Optional
import numpy as np
import pandas as pd
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain.schema import Document
from huggingface_hub import login

from config import CONFIG

logger = logging.getLogger(__name__)

login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))


class EmbeddingManager:
    """Manager for creating and handling embeddings."""
    
    def __init__(self, model_name: str = CONFIG.EMBEDDING_MODEL):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name of the HuggingFace model to use for embeddings
        """
        logger.info(f"Initializing embedding manager with model: {model_name}")
        
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        self.vector_store = None
    
    def create_embeddings_from_dataframe(
        self, 
        df: pd.DataFrame,
        text_column: str = "combined_text",
        metadata_columns: List[str] = ["query_id", "response_id", "query", "response"],
        use_combined: bool = True,
        vector_db_type: str = "faiss"
    ) -> Union[FAISS, Chroma]:
        """
        Create embeddings from a DataFrame and store them in a vector database.
        
        Args:
            df: DataFrame containing text data
            text_column: Column to use for generating embeddings
            metadata_columns: Columns to include as metadata
            use_combined: Whether to use combined query+response text
            vector_db_type: Type of vector database to create ("faiss" or "chroma")
            
        Returns:
            Vector store object
        """
        logger.info(f"Creating embeddings from DataFrame with {len(df)} entries")
        
        # Document creation
        documents = []
        for _, row in df.iterrows():

            if use_combined:
                content = row[text_column]
            else:
                # Use only query for embedding(added thsi so as to check in future)
                content = row["query_clean"]
            
            metadata = {col: row[col] for col in metadata_columns if col in row}
            
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        
        logger.info(f"Created {len(documents)} documents for embedding")
        

        if vector_db_type.lower() == "faiss":
            vector_store = FAISS.from_documents(documents, self.embedding_model)
            logger.info("Created FAISS vector store")
        elif vector_db_type.lower() == "chroma":

            os.makedirs(CONFIG.VECTOR_DB_PATH, exist_ok=True)
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                persist_directory=str(CONFIG.VECTOR_DB_PATH)
            )
            vector_store.persist()
            logger.info(f"Created Chroma vector store and persisted to {CONFIG.VECTOR_DB_PATH}")
        else:
            raise ValueError(f"Unsupported vector database type: {vector_db_type}")
        
        self.vector_store = vector_store
        return vector_store
    
    def load_vector_store(self, vector_db_type: str = "faiss", path: Optional[str] = None) -> Union[FAISS, Chroma]:
        """
        Load an existing vector store.
        
        Args:
            vector_db_type: Type of vector store ("faiss" or "chroma")
            path: Path to the vector store
            
        Returns:
            Loaded vector store
        """
        path = path or CONFIG.VECTOR_DB_PATH
        
        logger.info(f"Loading {vector_db_type} vector store from {path}")
        
        if vector_db_type.lower() == "faiss":
            vector_store = FAISS.load_local(path, self.embedding_model,allow_dangerous_deserialization=True)
            logger.info("Loaded FAISS vector store")
        elif vector_db_type.lower() == "chroma":
            vector_store = Chroma(
                persist_directory=str(path),
                embedding_function=self.embedding_model
            )
            logger.info("Loaded Chroma vector store")
        else:
            raise ValueError(f"Unsupported vector database type: {vector_db_type}")
        
        self.vector_store = vector_store
        return vector_store
    
    def get_similar_documents(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve similar documents from the vector store.
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            
        Returns:
            List of similar documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_embeddings_from_dataframe or load_vector_store first.")
        
        logger.info(f"Retrieving {k} similar documents for query: {query}")
        
        docs = self.vector_store.similarity_search(query, k=k)
        
        logger.info(f"Retrieved {len(docs)} documents")
        return docs
    
    def get_similar_documents_with_scores(self, query: str, k: int = 3) -> List[tuple]:
        """
        Retrieve similar documents with similarity scores.
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_embeddings_from_dataframe or load_vector_store first.")
        
        logger.info(f"Retrieving {k} similar documents with scores for query: {query}")
        
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
        
        logger.info(f"Retrieved {len(docs_with_scores)} documents with scores")
        return docs_with_scores
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get the embedding for a piece of text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self.embedding_model.embed_query(text)


if __name__ == "__main__":
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    

    from data.data_loader import load_or_create_processed_data
    
    manager = EmbeddingManager()
    
    df = load_or_create_processed_data()
    

    sample_df = df.sample(n=1000, random_state=42)

    # Create embeddings
    vector_store = manager.create_embeddings_from_dataframe(
        sample_df,
        vector_db_type="faiss"
    )

    # for full data
    # Create embeddings
    # vector_store = manager.create_embeddings_from_dataframe(
    #     df,
    #     vector_db_type="faiss"
    # )

    os.makedirs(CONFIG.DATA_DIR, exist_ok=True)

    vector_store.save_local(CONFIG.VECTOR_DB_PATH)  
    
    # Testing retrieval
    query = "I need help with my password reset"
    docs = manager.get_similar_documents(query, k=3)
    
    print("\nSimilar documents:")
    for i, doc in enumerate(docs):
        print(f"\nDocument {i+1}:")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")
    
    # Testing retrieval with scores
    docs_with_scores = manager.get_similar_documents_with_scores(query, k=3)
    
    print("\nSimilar documents with scores:")
    for i, (doc, score) in enumerate(docs_with_scores):
        print(f"\nDocument {i+1}:")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")
        print(f"Similarity Score: {score}")
