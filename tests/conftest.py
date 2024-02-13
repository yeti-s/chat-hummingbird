import pytest
import os
from dotenv import load_dotenv

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

@pytest.fixture
def open_api_key():
    return OPENAI_API_KEY
@pytest.fixture
def generator_model():
    return GENERATOR_MODEL
@pytest.fixture
def embedding_model():
    return EMBEDDING_MODEL
@pytest.fixture
def db_host():
    return DB_HOST
@pytest.fixture
def db_port():
    return DB_PORT
@pytest.fixture
def summarizer_model():
    return SUMMARIZER_MODEL