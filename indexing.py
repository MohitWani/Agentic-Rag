from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
import os
import pprint

def load_document(path):
    loader = PyPDFDirectoryLoader(path)
    documents = loader.load()

    print('loading is completed...')
    return documents

def splitter(document):
    doc_split = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap = 200,
    )

    docs_metadata = doc_split.split_documents(documents=document)

    docs = [doc for doc in docs_metadata]
    print("Chunks are created successfully...")
    return docs
    

def create_vectorstore(docs, embedding=GPT4AllEmbeddings(), path_tosave='D:\my_projects\oneture assignment'):

    vectorstore = FAISS.from_documents(docs, embedding)

    path = path_tosave
    folder = "faiss_index"

    persist_db = os.path.join(path, folder)
    os.makedirs(persist_db, exist_ok=True)

    vectorstore.save_local(persist_db)
    return "Vector database is Saved."

if __name__=="__main__":
    embedding = GPT4AllEmbeddings()

    document = load_document('D:\my_projects\oneture assignment\docs')
    split = splitter(document=document)
    print(len(split))
    create_vectorstore(split, embedding)