import os
from typing import List, Any
import numpy as np
import uuid
import chromadb


class VectorStore:
    """
    Gère le stockage et la récupération de vecteurs d'embeddings.
    """

    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "../data/vector_store"):
        """
        Initialise le magasin de vecteurs.

        Args:
            collection_name (str): Le nom de la collection de documents. Par défaut : "pdf_documents".
            persist_directory (str): Le répertoire où les données seront persistées. Par défaut : "../data/vector_store".
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialiser_magasin()

    def _initialiser_magasin(self):
        """
        Initialise le magasin de vecteurs en créant le répertoire s'il n'existe pas et en chargeant la collection.
        """
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Embeddings de documents PDF pour le RAG"}
            )

            print(f"Magasin de vecteurs initialisé. Collection : {self.collection_name}")
            print(f"Documents existants dans la collection : {self.collection.count()}")
        except Exception as e:
            print(f"Erreur lors de l'initialisation du magasin de vecteurs : {e}")
            raise

    def ajouter_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Ajoute des documents et leurs embeddings au magasin de vecteurs.

        Args:
            documents (List[Any]): Une liste de documents.
            embeddings (np.ndarray): Un tableau NumPy contenant les embeddings correspondants.

        Raises:
            ValueError: Si le nombre de documents ne correspond pas au nombre d'embeddings.
        """
        if len(documents) != len(embeddings):
            raise ValueError("Le nombre de documents doit correspondre au nombre d'embeddings.")

        print(f"Ajout de {len(documents)} documents au magasin de vecteurs...")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)

            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Documents ajoutés avec succès au magasin de vecteurs")
            print(f"Nombre total de documents dans la collection : {self.collection.count()}")

        except Exception as e:
            print(f"Erreur lors de l'ajout des documents au magasin de vecteurs : {e}")
            raise


vectorstore = VectorStore()
