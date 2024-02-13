from typing import Union, Callable

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    HumanMessagePromptTemplate, 
    SystemMessagePromptTemplate,
    ChatPromptTemplate
)

from generator.generator import Generator, CustomCallbackHandler

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
    








############################################################################################
######################################## TEST CODE #########################################
############################################################################################



def test_open_ai(model_name, openai_api_key):
    print('===== TEST GENERATION & CALLBACK HANDLER =====')
    on_llm_new_sentence_handler = lambda sentence: print(f'sentence complete! : {sentence}')
    on_llm_end_handler = lambda result: print(f'generation complete! : {result}')
    generator = OpenAIGenerator(model_name, openai_api_key, debug=True)
    
    print('===== GENERATE WITHOUT SUMMARY =====')
    query = "나는 인센, 안녕 토니 잘 지냈니?"
    persona = "나는 천재 공학자이자 스타크 인더스트리의 CEO 토니 스타크, 아프가니스탄에서 자기가 개발한 신무기 '제리코' 미사일[1]을 홍보하고 기지로 귀환 중에 테러리스트들의 습격을 받고 치명상을 입은 채 납치당한다.[2] 하필이면 테러리스트들의 습격 당시 심장에 미사일 파편이 박혀 곧 죽을 위기에 처하지만,[3] 테러리스트들에게 잡혀 있던 흉부외과 의사 호 인센이 제거할 수 있는 파편은 다 제거하고, 심장 부근에 전자석을 심어 미처 제거할 수 없었던 파편이 심장으로 유입되는 것을 막은 덕분에 겨우 목숨을 건진다.[4] 토니를 납치한 테러 단체 텐 링즈의 수장인 라자는 토니에게 제리코 미사일을 만들면 풀어주겠다고 제안하지만, 토니는 동굴에서 수작업으로 첨단 미사일을 만들어내라는 황당한 제안에 당연히 거부감을 느끼고, 응해 보았자 토사구팽당할 것이라고 여기며 절망한다. 그러나 인센의 동기부여로 토니 스타크는 인센과 함께 치밀하게 작전을 세워 빠져나갈 궁리를 하게 되고, 무기를 만드는 척하면서 공수 받은 재료들과 미사일 부품을 뜯어낸 것으로 팔라듐 전지와 급조 동력장갑복을 개발하여 탈출을 시도한다. 그러나 탈출 과정에서 인센은 슈트의 부팅 시간을 벌기 위해 테러리스트들을 유인하다가 사망하고,[5] 인센의 희생에 크게 분노한 토니는 Mk.1 슈트에 달린 무기들로 테러리스트들을 한바탕 쓸어버리고 무사히 탈출한다."
    print(f'query : {query}')
    print(f'persoan : {persona}')
    generator.generate(
        query, 
        persona,
        on_llm_new_sentence_handler=on_llm_new_sentence_handler,
        on_llm_end_handler=on_llm_end_handler
    )
    
    print('===== GENERATE WITH SUMMARY =====')
    query = "나는 인센, 우리가 언제 처음 만난곳이 아프가니스탄이었지?"
    summary = "인센과 스타크의 대화. 서로 안부 인사를 물어보았다."
    print(f'query : {query}')
    print(f'summary: {summary}')
    generator.generate(
        query, 
        persona,
        summary,
        on_llm_new_sentence_handler=on_llm_new_sentence_handler,
        on_llm_end_handler=on_llm_end_handler
    )
    print('===== FINISH GENERATION & CALLBACK HANDLER =====')
    