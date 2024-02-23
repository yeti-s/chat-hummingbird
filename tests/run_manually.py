
import argparse

from conftest import DB_HOST, DB_PORT, EMBEDDING_MODEL, GENERATOR_MODEL, OPENAI_API_KEY, SUMMARIZER_MODEL
from test_chatbot import run_chatbot
from test_chroma_manager import run_chroma_manager
from test_open_ai_generator import run_open_ai_generator
from test_summarizer import run_summarizer

if __name__ == '__main__':
    print('----- ENVIRONMENT -----')
    print(f'DB_HOST: {DB_HOST}')
    print(f'OPENAI_API_KEY: {OPENAI_API_KEY}')
    print(f'GENERATOR_MODEL: {GENERATOR_MODEL}')
    print(f'EMBEDDING_MODEL: {EMBEDDING_MODEL}')
    print(f'DB_PORT: {DB_PORT}')
    print(f'SUMMARIZER_MODEL: {SUMMARIZER_MODEL}')
    print('------------------------')
    
    parser = argparse.ArgumentParser(description='test modules manually.')
    parser.add_argument('--chatbot', action='store_true', help='test chatbot module.')
    parser.add_argument('--chroma', action='store_true', help='test chroma manager module.')
    parser.add_argument('--openai', action='store_true', help='test open ai generator module.')
    parser.add_argument('--summarizer', action='store_true', help='test summarizer module.')
    
    args = parser.parse_args()
    
    if args.summarizer:
        run_summarizer(SUMMARIZER_MODEL)
        
    if args.chroma:
        run_chroma_manager(DB_HOST, DB_PORT, EMBEDDING_MODEL)
        
    if args.openai:
        run_open_ai_generator(GENERATOR_MODEL, OPENAI_API_KEY)
    
    if args.chatbot:
        run_chatbot(DB_HOST, DB_PORT, EMBEDDING_MODEL, GENERATOR_MODEL, OPENAI_API_KEY, SUMMARIZER_MODEL)
    