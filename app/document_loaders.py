import os
from dotenv import load_dotenv
import tempfile
from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader
)
import glob

load_dotenv()

def load_text_file():

    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
        temp_file.write(b"Hello, this is a sample text file.\\nThis file is used to ...")
        temp_file_path = temp_file.name

    try:
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        for doc in documents:
            print("Document content: ")
            print(doc)
            print(doc.page_content)

    finally:
        os.remove(temp_file_path)


def pdf_loader():
    
    all_documents = []
    for path in glob.glob('docs/**/*.pdf', recursive=True):
        loader = PyPDFLoader(path)
        documents = loader.load()

        all_documents.extend(documents)

    print(f'Loaded {len(all_documents)} documents')

    return all_documents

# if __name__ == '__main__':
#     pdf_loader('C:\\Users\\ACER\\Desktop\\AI101-RAG\\docs\\KKTC_RENC_Z_NLER_S_STEM_TRNC_STUDENT_PERMISSION_SYSTEM.pdf')
# #     load_text_file()