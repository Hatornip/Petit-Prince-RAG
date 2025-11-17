import os
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

def process_all_pdfs(pdf_directory):
    all_documents = []
    pdf_dir = Path(pdf_directory)

    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files to process")

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'

            all_documents.extend(documents)
            print(f"Loaded {len(documents)} pages")

        except Exception as e:
            print(f"Error {e}")
        print(f"Total documents loaded: {len(all_documents)}")
        return all_documents


def split_documents(documents,chunk_size=1000,chunk_overlap=200):
    text_splitter=RecursiveCharacterTextSplitter(
       chunk_size = chunk_size,
       chunk_overlap=chunk_overlap,
       length_function=len,
       separators=["\n\n","\n"," ",""] 
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content: {split_docs[1].page_content[:200]}...")
        print(f"Metadata: {split_docs[1].metadata}")
    return split_docs
