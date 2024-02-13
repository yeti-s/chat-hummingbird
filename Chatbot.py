from typing import Union

from vectordb.db_manager import DBManager
from generator.generator import Generator
from summarizer.summarizer import Summarizer


class Chatbot():
    def __init__(self, generator:Generator, summarizor:Summarizer, db_manager:DBManager) -> None:
        self.db_manager = db_manager
        self.generator = generator
        self.summarizor = summarizor
    
    def generate(self, query:str, user_id:str, summary:Union[None, str]=None) -> tuple[str, str]:
        persona = self.db_manager.search_persona(query, user_id)
        generated = self.generator.generate(query, persona, summary)
        new_summary = self.summarizor.summarize([query, generated.content])
        return generated, new_summary