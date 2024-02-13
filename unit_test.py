import os
from dotenv import load_dotenv

from summarizer.summarizer import test_summarizer
from vectordb.chroma_manager import test_chroma_manager
from generator.openai import test_open_ai

def test_all_unit():
    
    print('==== UNIT TEST START')
    
    load_dotenv()
    # Generator
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    GENERATOR_MODEL = os.getenv('GENERATOR_MODEL')
    # DB Manager
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    # Summarizer
    SUMMARIZER_MODEL = os.getenv('SUMMARIZER_MODEL')
    
    test_open_ai(GENERATOR_MODEL, OPENAI_API_KEY)
    test_chroma_manager(DB_HOST, DB_PORT, EMBEDDING_MODEL)
    test_summarizer(SUMMARIZER_MODEL)
    
    

if __name__ == "__main__":
    test_all_unit()