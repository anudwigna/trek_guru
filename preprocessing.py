from dotenv import load_dotenv
from dotenv import dotenv_values


from dotenv import load_dotenv
from dotenv import dotenv_values
load_dotenv() 

CONFIG = dotenv_values(".env")
INPUT_DATA_PATH = './data/trekking.txt'
EMBEDDINGS_PATH = './data/embeddings.joblib'
VECTOR_STORE_PATH = './data/vectors.joblib'
INPUT_URL = "https://en.wikivoyage.org/wiki/Trekking_in_Nepal"

class Preprocessing():

    def __init__(self) -> None:
        pass
