import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from backend.config import PINECONE_API_KEY
    
# set environment variable for pinecone 
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# initalize pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

#define embedding models
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

INDEX_NAME="rag-index"

# retrieve function
def get_retriver():
    """Initialize and returns Pinecone vectorstore retriever"""
    # ensure index exists, create if not
    if INDEX_NAME not in pc.list_indexes().names():
        print("Creating new index.....")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws',region='us-east-1')
        )
        print("Created pinecone index successfully")
    vectorstore=PineconeVectorStore(index_name=INDEX_NAME,embedding=embeddings)
    return vectorstore.as_retriever()



# upload documents to vectorstore
def add_document(text_content:str):
    """
    Add a single text document to the pinecone vector store.
    Splits the text into chunks before embedding and upserting.
    """
    if not text_content:
        raise ValueError("Document content cannot be empty!")

    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    # create langchain document objects from the raw text
    documents = text_splitter.create_documents([text_content])
    print("Splitting documents into chunks for indexing...")

    # get vectorstore instance to add document
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    # add documents to vectorstore
    vectorstore.add_documents(documents)
    print("Successfully added chunks to pinecone vectorstore") 
