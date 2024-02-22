from typing import Union, Callable

from chat_hummingbird.vectordb.db_manager import DBManager
from chat_hummingbird.generator.generator import Generator
from chat_hummingbird.summarizer.summarizer import Summarizer

class Chatbot():
    def __init__(self, generator:Generator, summarizer:Summarizer, db_manager:DBManager) -> None:
        self.db_manager = db_manager
        self.generator = generator
        self.summarizer = summarizer
    
    def __get_dialogue_from_history__(self, history:list[tuple[str, str]]) -> list[str]:
        dialouge = []
        for turn in history:
            dialouge.append(turn[0])
            dialouge.append(turn[1])
        
        return dialouge
    
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
        example = None
        
        if history is not None:
            history_summary = self.summarizer.summarize(
                self.__get_dialogue_from_history__(history)
            )
            print(f'history_summary: {history_summary}')
            example = self.db_manager.search_message(
                history_summary,
                relation
            )
        
        generated = self.generator.generate(
            user_name,
            ai_name,
            query, 
            persona,
            relation,
            example,
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