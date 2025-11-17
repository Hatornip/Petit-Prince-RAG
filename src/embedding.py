import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List,Dict,Any,Tuple
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingManager:
    """
    Manages the loading and generation of embeddings from a SentenceTransformer model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the embedding manager.

        Args:
            model_name (str): The name of the SentenceTransformer model to use. 
                              Defaults to "all-MiniLM-L6-v2".
        """
        self.model_name = model_name
        self.model = None  # Initialize to None to indicate that the model is not yet loaded

        print(f"Initializing embedding manager with model: {self.model_name}") # More informative message during initialization
        self._load_model()


    def _load_model(self):
        """
        Loads the specified SentenceTransformer model.

        Raises:
            Exception: If loading the model fails.
        """
        try:
            print(f"Loading embedding model: {self.model_name}...")  # Clearer message
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}") #Displaying the dimension
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")  # More detailed error message
            raise  # Re-raise the exception to signal failure


    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of texts.

        Args:
            texts (List[str]): A list of strings to be converted into embeddings.

        Returns:
            np.ndarray: A NumPy array containing the generated embeddings.

        Raises:
            ValueError: If the model has not been loaded.
        """
        if not self.model:
            raise ValueError("Model not loaded. Please initialize the embedding manager.") #Clearer message if there is an issue
        
        print(f"Generating embeddings for {len(texts)} texts...") #More descriptive
        try:
          embeddings = self.model.encode(texts, show_progress_bar=True)  # Keeping progress bar
          print(f"Embeddings generated with shape: {embeddings.shape}")
        except Exception as e:
            print(f"Error generating embeddings: {e}") #More detailed error message if there is an issue
            raise

        return embeddings
