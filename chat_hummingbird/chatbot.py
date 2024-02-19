from typing import Union, Callable

from chat_hummingbird.vectordb.db_manager import DBManager
from chat_hummingbird.generator.generator import Generator
from chat_hummingbird.summarizer.summarizer import Summarizer

class Chatbot():
    def __init__(self, generator:Generator, summarizer:Summarizer, db_manager:DBManager) -> None:
        self.db_manager = db_manager
        self.generator = generator
        self.summarizer = summarizer
    
    def generate(
        self, 
        user_name:str,
        ai_name:str,
        query:str, 
        user_id:str,
        relation:str='지인',
        summary:Union[None, str]=None,
        history:Union[None, list[tuple[str, str]]]=None,
        on_llm_new_sentence_handler:Union[None, Callable]=None,
        on_llm_end_handler:Union[None, Callable]=None,
        on_llm_error_handler:Union[None, Callable]=None
    ) -> tuple[str, str]:
        persona = self.db_manager.search_persona(query, user_id)
        generated = self.generator.generate(
            user_name,
            ai_name,
            query, 
            persona,
            relation,
            summary,
            history,
            on_llm_new_sentence_handler=on_llm_new_sentence_handler,
            on_llm_end_handler=on_llm_end_handler,
            on_llm_error_handler=on_llm_error_handler
        )
        
        dialogues = []
        if summary is not None and history is not None:
            dialogues = [summary]
            for turn in history:
                dialogues.extend(turn)
        dialogues.extend([query, generated])
        summary = self.summarizer.summarize(dialogues)
        
        return generated, summary