from vectorstore import VectorStore
from embedding import EmbeddingManager
from typing import List, Any,Dict

class RAGRetriever:

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialise le récupérateur de documents pour la recherche augmentée par le contexte (RAG).

        Args:
            vector_store (VectorStore): L'objet magasin de vecteurs contenant les embeddings des documents.
            embedding_manager (EmbeddingManager): L'objet gérant l'encodage en vecteurs des textes.
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def rechercher(self, requete: str, top_k: int = 5, seuil_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Récupère les documents les plus pertinents pour une requête donnée en utilisant la recherche vectorielle et un filtrage par score.

        Args:
            requete (str): La requête de l'utilisateur.
            top_k (int): Le nombre maximum de documents à renvoyer. Par défaut : 5.
            seuil_score (float): Le seuil de similarité à appliquer pour filtrer les résultats. Par défaut : 0.0.

        Returns:
            List[Dict[str, Any]]: Une liste de dictionnaires contenant les informations sur les documents récupérés, triés par pertinence décroissante.  Chaque dictionnaire contient l'ID du document, le contenu, les métadonnées associées, le score de similarité, la distance et son rang dans les résultats.
        """

        print(f"Recherche de documents pour la requête : {requete}")
        print(f"Top K : {top_k}, Seuil de score : {seuil_score}")

        # Générer l'embedding vectoriel de la requête
        embedding_requete = self.embedding_manager.generate_embeddings([requete])[0]

        try:
            # Effectuer la requête vectorielle dans le magasin de vecteurs
            resultats = self.vector_store.collection.query(
                query_embeddings=[embedding_requete.tolist()],  # Convertir en liste pour la compatibilité avec chromadb
                n_results=top_k
            )

            documents_trouves = []

            # Vérifier si des documents ont été trouvés et s'ils sont présents
            if resultats['documents'] and resultats['documents'][0]:
                documents = resultats['documents'][0]
                metadonnees = resultats['metadatas'][0]
                distances = resultats['distances'][0]
                ids = resultats['ids'][0]

                # Itérer sur les résultats et filtrer par score de similarité
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadonnees, distances)):
                    score_similarite = 1 - distance  # Calculer le score de similarité à partir de la distance

                    if score_similarite >= seuil_score:
                        documents_trouves.append({
                            'id': doc_id,
                            'contenu': document,
                            'metadonnees': metadata,
                            'score_similarite': score_similarite,
                            'distance': distance,
                            'rang': i + 1  # Ajouter le rang dans les résultats
                        })

            else:
                print("Aucun document trouvé.")

            print(f"Récupération de {len(documents_trouves)} documents après filtrage.")
            return documents_trouves

        except Exception as e:
            print(f"Erreur lors de la recherche : {e}")
            return []


