from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from preprocessing import CONFIG
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
import joblib
from preprocessing import VECTOR_STORE_PATH

class Inference():

    llm: HuggingFaceHub
    chain: BaseCombineDocumentsChain
    
    def __init__(self) -> None:
        self.llm = HuggingFaceHub(
            repo_id="google/flan-t5-xxl", 
            model_kwargs={"temperature":0.5, "max_length":1024}, 
            huggingfacehub_api_token=CONFIG["HUGGINGFACE_TOKEN"]
        )
        self.chain = load_qa_chain(self.llm, chain_type="stuff")

    def predict_answer(self, query: str) -> str:
        the_db = joblib.load(VECTOR_STORE_PATH)
        docs = the_db.similarity_search(query)
        return self.chain.run(input_documents=docs, question=query)

    