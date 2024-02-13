from typing import Any, Union, Callable
from abc import ABCMeta, abstractmethod

from langchain.callbacks import StdOutCallbackHandler
from langchain_core.outputs import LLMResult

class CustomCallbackHandler(StdOutCallbackHandler):
    def __init__(
        self,
        on_llm_new_token_handler:Union[None, Callable]=None,
        on_llm_end_handler:Union[None, Callable]=None,
        on_llm_error_handler:Union[None, Callable]=None
    ) -> None:
        super().__init__()
        self.on_llm_new_token_handler = on_llm_new_token_handler
        self.on_llm_end_handler = on_llm_end_handler
        self.on_llm_error_handler = on_llm_error_handler
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        if self.on_llm_new_token_handler is not None:
            self.on_llm_new_token_handler(token)
            
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        if self.on_llm_end_handler is not None:
            self.on_llm_end_handler(response)
            
    def on_llm_error(self, error: BaseException, **kwargs: Any) -> Any:
        # return super().on_llm_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        pass

class Generator(metaclass=ABCMeta):
    def __init__(
        self, 
        on_llm_new_sentence_handler:Union[None, Callable]=None,
        on_llm_end_handler:Union[None, Callable]=None,
        on_llm_error_handler:Union[None, Callable]=None
    ) -> None:
        
        self.buffers = []
        self.on_llm_new_sentence_handler = on_llm_new_sentence_handler
        self.callback_handler = CustomCallbackHandler(
            on_llm_new_token_handler=self.on_llm_new_token_handler,
            on_llm_end_handler=on_llm_end_handler,
            on_llm_error_handler=on_llm_error_handler
        )

    def on_llm_new_token_handler(self, token:str) -> None:
        if self.on_llm_new_sentence_handler is None:
            return
        
        self.buffers.append(token)
        if token in ['.', '\n', '.\n', '\n\n']:
            self.on_llm_new_sentence_handler(''.join(self.buffers))
            self.buffers = []
            

    @abstractmethod
    def generate(self, query, persona:str, summary:Union[None, str]=None) -> str:
        pass
    
