from preprocessing import INPUT_DATA_PATH, EMBEDDINGS_PATH, VECTOR_STORE_PATH

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import joblib

class Train():

    def __init__(self) -> None:
        pass

    def start_training(self):
        loader = TextLoader(INPUT_DATA_PATH, encoding='utf-8')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings() 
        joblib.dump(embeddings, EMBEDDINGS_PATH)
        db = FAISS.from_documents(docs, embeddings)
        joblib.dump(db, VECTOR_STORE_PATH)



