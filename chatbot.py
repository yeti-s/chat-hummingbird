from typing import Union, Callable

from vectordb.db_manager import DBManager
from generator.generator import Generator
from summarizer.summarizer import Summarizer

class Chatbot():
    def __init__(self, generator:Generator, summarizor:Summarizer, db_manager:DBManager) -> None:
        self.db_manager = db_manager
        self.generator = generator
        self.summarizor = summarizor
    
    def generate(
        self, 
        query:str, 
        user_id:str, 
        summary:Union[None, str]=None,
        on_llm_new_sentence_handler:Union[None, Callable]=None,
        on_llm_end_handler:Union[None, Callable]=None,
        on_llm_error_handler:Union[None, Callable]=None
    ) -> tuple[str, str]:
        persona = self.db_manager.search_persona(query, user_id)
        generated = self.generator.generate(
            query, 
            persona, 
            summary,
            on_llm_new_sentence_handler=on_llm_new_sentence_handler,
            on_llm_end_handler=on_llm_end_handler,
            on_llm_error_handler=on_llm_error_handler
        )
        new_summary = self.summarizor.summarize([query, generated] if summary is None else [summary, query, generated])
        return generated, new_summary