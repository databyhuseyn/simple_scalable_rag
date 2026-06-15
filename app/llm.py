from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def get_llm():

    llm = ChatOpenAI(
            model = os.getenv("MODEL_NAME"), 
            api_key = os.getenv("OPENAI_API_KEY"),
            base_url = os.getenv("BASE_URL")
        )

    return llm