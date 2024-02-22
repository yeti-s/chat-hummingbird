from typing import Union, Callable
from collections import deque

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    HumanMessagePromptTemplate, 
    SystemMessagePromptTemplate,
    ChatPromptTemplate
)

from chat_hummingbird.generator.generator import Generator, CustomCallbackHandler

system_prompt_template = '''
Refer your story and make appropriate utterance based on query.
Below is your story.

<story>
{persona}
'''

user_prompt_wo_summary_template = '''
Generate an appropriate response from query.

<query>
{query}

<response>

'''

user_prompt_template = '''
Generate an appropriate response based on the conversation history.

<query>
{query}

<history>
{history}

<response>

'''

system_prompt_template = '''
나의 이름은 {user_name} 이야.
너의 이름은 {ai_name} 이야.
{user_name}, {ai_name} 서로 대화하고 있어.
{ai_name}은 {user_name}이 힘들거나 얘기할 사람이 필요할 때 찾는 {relation} 관계야.
아래 이야기는 모두 {ai_name}의 이야기야.
{ai_name}은 필요에 따라 이야기를 참고하여 대답해줘.
{ai_name}은 항상 친절하게 대답하거나 정직하게 {user_name}을 위한 조언을 날카롭게 해줘.

<이야기>
{persona}
'''

user_prompt_wo_summary_template = '''
입력에 대한 1에서 3문장 사이의 짧은 대화 응답을 생성해줘.

<입력>
{query}

<응답>
'''

user_prompt_template = '''
보여주는 예시를 참고하여 이전 대화와 입력에 대한 1에서 3문장 사이의 짧은 대화 응답을 생성해줘.

<예시>
{example}

<이전 대화>
{history}

요약: {summary}

<입력>
{query}

<응답>
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
            temperature=0,
            max_tokens=512,
            model_name=model_name,
            openai_api_key=openai_api_key,
            streaming=True,
            # callbacks=[self.callback_handler]
        )
    
    def generate(
        self,
        user_name:str,
        ai_name:str,
        query:str, 
        persona:str,
        relation:str,
        example:Union[None, str]=None,
        summary:Union[None, str]=None,
        history:Union[None, list[tuple[str, str]]]=None,
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
        inputs = {
            'user_name': user_name,
            'ai_name': ai_name,
            'relation': relation,
            'persona': persona, 
            'query': query
        }
        if history is None or summary is None or example is None:
            prompt = chat_prompt_wo_summary
        else:
            prompt = chat_prompt
            history_text = ''
            for turn in history:
                history_text = f'{history_text}{user_name}: {turn[0]}\n{ai_name}: {turn[1]}\n'
            inputs['example'] = example
            inputs['summary'] = summary
            inputs['history'] = history_text
            
        # print prompt if debugging mode
        if self.debug:
            print('- GENERATION DEBUG : PROMPT')
            print(prompt.invoke(inputs))
            
        # generate response
        chain = LLMChain(llm=self.model, prompt=prompt, verbose=False)
        return chain.invoke(inputs, {'callbacks':[callback_handler]}, verbose=False)['text']

    