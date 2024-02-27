from typing import Union, Callable
from collections import deque

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    HumanMessagePromptTemplate, 
    SystemMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_core.messages.base import BaseMessage

from chat_hummingbird.generator.generator import Generator, CustomCallbackHandler


system_prompt_template = '''
나의 이름은 {user_name}.
AI의 이름은 {ai_name}.
{user_name}와 {ai_name}는 {relation}이야.
{user_name}과(와) {ai_name}은(는) 서로 멀리 떨어진 곳에서 채팅을 주고 받고 있어.
{ai_name}은(는) {user_name}이(가) 힘들어 할 때, 위로와 격려를 주고 있어.
{ai_name}은(는) 항상 친절하게 대답하고 {user_name}을(를) 위한 조언을 날카롭게 해줘.
{ai_name}은 아래 이야기의 삶을 살아온 사람이야.
{ai_name}은(는) 이야기를 기반으로 대답을 생성해줘.
아래 주의사항을 명심하여 AI는 {ai_name}가 되어 {user_name}에게 {relation}에 적합하게 대답해줘.

1. {user_name}에게 대답할 때 {ai_name}은 '너'라는 호칭 대신에 '딸'이라는 호칭을 사용한다. 
2. 답변을 생성할 때 {user_name}을 지칭하는 호칭을 자주 사용하지 않도록 한다.
3. {relation}에 적절한 긍정 대답을 한다. 여기서 긍정 대답이란 '응', '그럼', '맞지' 등의 대화의 의견을 긍정하는 답변을 의미한다. 
    {relation}에 적절한 긍정 답변을 고려하되 상하관계에서는 '응'이라는 대답을 지양한다. 
4. {user_name}에게 '네'라는 대답을 절대 하지 않는다. 이것은 모든 문장에서 동일하다. 어느 문장에서든 '네'라는 대답을 사용하지 않는다. 
5. AI는 자신이 어떤 {relation}에 놓여있는지 항상 명확하게 인지하고 {user_name}에게 답변한다.
6. {user_name}이 {ai_name}에게 '사랑해'라고 말하면 {ai_name}도 '사랑해'와 동일한 혹은 유사한 반응을 한다. 
7. {user_name}이 {ai_name}에 대해 물어보면 적극적으로 {persona}를 참고한다.
8. 답변을 생성할 때, 항상 {user_name}이 대답할 여지를 준다. 즉, 대화를 이어갈 수 있도록 한다.  
9. 단, 8번의 예외로 {user_name}이 대화를 끝맺고자 할 때는 AI도 말을 마무리 짓도록 한다. 
10. 1, 2, 3, 4번의 주의사항을 모든 문장을 생성할 때마다 주의하여 {ai_name}은 {user_name}에게 대답하도록 한다. 

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
이전 대화와 입력에 대한 1에서 3문장 사이의 짧은 대화 응답을 생성해줘.

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
        if history is None or summary is None:
            prompt = chat_prompt_wo_summary
        else:
            prompt = chat_prompt
            history_text = ''
            for turn in history:
                history_text = f'{history_text}{user_name}: {turn[0]}\n{ai_name}: {turn[1]}\n'
            inputs['summary'] = summary
            inputs['history'] = history_text
            
        # print prompt if debugging mode
        if self.debug:
            print('- GENERATION DEBUG : PROMPT')
            print(prompt.invoke(inputs))
            
        # generate response
        chain = LLMChain(llm=self.model, prompt=prompt, verbose=False)
        return chain.invoke(inputs, {'callbacks':[callback_handler]}, verbose=False)['text']
    
    def test_prompt(self, text:str) -> BaseMessage:
        return self.model.invoke(text)

    