from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings











#Extract text from PDF files
def load_pdf_files(data):
    loader=DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents=loader.load()
    return documents



#filter the doc and remove unnecessary metadata
def filter_to_minimal_docs(docs:List[Document],min_length:int=1000)->List[Document]:
    """
    Given a list of Documents objects, return a new list of Document objects containing only 'source' in metadata
    and the original page_content
    """
    minimal_docs:List[Document]=[]
    for doc in docs:
        src=doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src}
            )
        )
    return minimal_docs




#split the data in smaller chunks
def text_split(minimal_docs):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    text_chunk=text_splitter.split_documents(minimal_docs)
    return text_chunk


#embedding model
def download_embeddings():
    """
    Dowmlaod and return the HuggingFace embedding model
    """
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    embeddings=HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings


