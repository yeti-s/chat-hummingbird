from typing import Union, Callable

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    HumanMessagePromptTemplate, 
    SystemMessagePromptTemplate,
    ChatPromptTemplate
)

from src.generator.generator import Generator, CustomCallbackHandler

system_prompt_template = '''
You are my friend.
Refer your story and make appropriate utterance based on query.
Below is your story.

<story>
{persona}
'''

user_prompt_wo_summary_template = '''
Generate an appropriate response from query.

<query>
{query}
'''

user_prompt_template = '''
Generate an appropriate response based on the conversation history.

<query>
{query}

<history>
{summary}
'''

system_template = SystemMessagePromptTemplate.from_template(system_prompt_template)
human_template = HumanMessagePromptTemplate.from_template(user_prompt_template)
human_template_wo_summary = HumanMessagePromptTemplate.from_template(user_prompt_wo_summary_template)
chat_prompt = ChatPromptTemplate.from_messages([system_template, human_template])
chat_prompt_wo_summary = ChatPromptTemplate.from_messages([system_template, human_template_wo_summary])

class OpenAIGenerator(Generator):
    def __init__(
        self, 
        model_name:str, 
        openai_api_key:str,
        debug:bool = False
    ) -> None:
        
        self.debug = debug
        self.model = ChatOpenAI(
            temperature=0.5,
            max_tokens=512,
            model_name=model_name,
            openai_api_key=openai_api_key,
            streaming=True,
            # callbacks=[self.callback_handler]
        )
        
    def generate(
        self, 
        query:str, 
        persona:str, 
        summary:Union[None, str]=None,
        on_llm_new_sentence_handler:Union[None, Callable]=None,
        on_llm_end_handler:Union[None, Callable]=None,
        on_llm_error_handler:Union[None, Callable]=None
    ) -> str:
        
        # create callback 
        callback_handler = CustomCallbackHandler(
            on_llm_new_sentence_handler=on_llm_new_sentence_handler,
            on_llm_end_handler=on_llm_end_handler,
            on_llm_error_handler=on_llm_error_handler
        )
        
        # create prompt
        inputs = {'persona': persona, 'query': query}
        if summary is None:
            prompt = chat_prompt_wo_summary
        else:
            prompt = chat_prompt
            inputs['summary'] = summary
            
        # print prompt if debugging mode
        if self.debug:
            print('- GENERATION DEBUG : PROMPT')
            print(prompt.invoke(inputs))
            
        # generate response
        chain = LLMChain(llm=self.model, prompt=prompt, callbacks=[callback_handler])
        return chain.invoke(inputs, {'callbacks':[callback_handler]})['text']

    