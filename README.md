# PetitPrince-RAG: Un système de Question-Réponse basé sur LangChain et Gemma 3B

## Description

Ce projet démontre l'implémentation d'un système Retrieval-Augmented Generation (RAG) utilisant LangChain, le modèle de langage Gemma 3B open source hébergé sur LM Studio, et ChromaDB comme base de données vectorielle.  L'objectif est de permettre une compréhension fine du texte "Le Petit Prince" et de répondre à des questions complexes basées sur son contenu.

## Fonctionnalités Principales

*   **Extraction et Indexation:** Le texte intégral de "Le Petit Prince" est extrait et divisé en morceaux pour créer des embeddings vectoriels.
*   **Base de Données Vectorielle (ChromaDB):**  Les embeddings sont stockés dans ChromaDB, permettant une recherche rapide par similarité sémantique.
*   **LangChain:** Utilisation de LangChain pour construire le pipeline RAG, incluant la gestion des documents, la création d'embeddings et l'interaction avec le LLM.
*   **Gemma 3B (LM Studio):**  Intégration du modèle Gemma 3B open source hébergé sur LM Studio pour générer des réponses contextuellement pertinentes aux questions posées.
*   **Interface Utilisateur:** Une interface simple permet de poser des questions sur "Le Petit Prince" et d'obtenir des réponses basées sur le texte.

## Prérequis

*   Python 3.8+
*   [LangChain](https://python.langchain.com/)
*   [ChromaDB](https://chromadb.com/)
*   [LM Studio](https://lmstudio.ai/) (pour exécuter Gemma 3B)

## Installation

1.  Cloner le dépôt : `git clone https://github.com/Hatornip/Petit-Prince-RAG.git`
2.  Créer un environnement virtuel : `python -m venv .venv`
3.  Activer l'environnement virtuel:
    *   Linux/macOS : `source .venv/bin/activate`
    *   Windows : `.venv\Scripts\activate`
4.  Installer les dépendances : `pip install -r requirements.txt`
5.  Télécharger le modèle Gemma 3B depuis LM Studio et placer le fichier dans le répertoire approprié (voir section Configuration).

## Configuration

*   **LM Studio:** Téléchargez et installez LM Studio ([https://lmstudio.ai/](https://lmstudio.ai/)). Téléchargez une version de Gemma 3B adaptée à votre matériel.
*   **Répertoire des modèles :** Modifiez le fichier `config.yaml` pour spécifier le chemin d'accès au modèle Gemma 3B téléchargé dans LM Studio.  Assurez-vous que le nom du modèle correspond exactement au nom utilisé par LangChain.
*  **ChromaDB:** La base de donnée ChromaDB est initialisée automatiquement lors du premier lancement. Vous pouvez configurer les paramètres de ChromaDB (comme la taille de la collection) dans le fichier `config.yaml`.

## Utilisation

1.  Dans "main.py" il vous faut modifier le prompt pour poser vos questions, donc ligne 44 prompt par défaut "Qui es le Petit Prince ?"
2.  Exécuter le script principal : `python main.py`
3.  Les réponses seront générées par Gemma 3B en utilisant les informations extraites du texte.
