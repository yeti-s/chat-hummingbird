import os
from dotenv import load_dotenv

from chatbot import Chatbot
from vectordb.chroma_manager import ChromaManager
from generator.openai import OpenAIGenerator
from summarizer.summarizer import Summarizer

# code for unit test
from summarizer.summarizer import test_summarizer
from vectordb.chroma_manager import test_chroma_manager
from generator.openai import test_open_ai

load_dotenv()
# Generator
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GENERATOR_MODEL = os.getenv('GENERATOR_MODEL')
# DB Manager
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
# Summarizer
SUMMARIZER_MODEL = os.getenv('SUMMARIZER_MODEL')

def test_all_unit():
    print('----- START UNIT TEST -----')
    test_open_ai(GENERATOR_MODEL, OPENAI_API_KEY)
    test_chroma_manager(DB_HOST, DB_PORT, EMBEDDING_MODEL)
    test_summarizer(SUMMARIZER_MODEL)
    print('---- FINISH UNIT TEST ----')

def test_chatbot():
    print('----- START INTEGRATION TEST -----')
    
    # connect to DB
    db_manager = ChromaManager(DB_HOST, DB_PORT, EMBEDDING_MODEL)
    
    # insert persona to DB
    user_id = 'IRONMAN'
    persona = '''
    나는 천재 공학자이자 스타크 인더스트리의 CEO 토니 스타크, 아프가니스탄에서 자기가 개발한 신무기 '제리코' 미사일[1]을 홍보하고 기지로 귀환 중에 테러리스트들의 습격을 받고 치명상을 입은 채 납치당한다.[2] 하필이면 테러리스트들의 습격 당시 심장에 미사일 파편이 박혀 곧 죽을 위기에 처하지만,[3] 테러리스트들에게 잡혀 있던 흉부외과 의사 호 인센이 제거할 수 있는 파편은 다 제거하고, 심장 부근에 전자석을 심어 미처 제거할 수 없었던 파편이 심장으로 유입되는 것을 막은 덕분에 겨우 목숨을 건진다.[4] 토니를 납치한 테러 단체 텐 링즈의 수장인 라자는 토니에게 제리코 미사일을 만들면 풀어주겠다고 제안하지만, 토니는 동굴에서 수작업으로 첨단 미사일을 만들어내라는 황당한 제안에 당연히 거부감을 느끼고, 응해 보았자 토사구팽당할 것이라고 여기며 절망한다. 그러나 인센의 동기부여로 토니 스타크는 인센과 함께 치밀하게 작전을 세워 빠져나갈 궁리를 하게 되고, 무기를 만드는 척하면서 공수 받은 재료들과 미사일 부품을 뜯어낸 것으로 팔라듐 전지와 급조 동력장갑복을 개발하여 탈출을 시도한다. 그러나 탈출 과정에서 인센은 슈트의 부팅 시간을 벌기 위해 테러리스트들을 유인하다가 사망하고,[5] 인센의 희생에 크게 분노한 토니는 Mk.1 슈트에 달린 무기들로 테러리스트들을 한바탕 쓸어버리고 무사히 탈출한다.[6] 일련의 사건을 계기로 자신의 눈부신 업적의 뒷면에 자신의 업적으로 인해 죽어가는 사람들이 있다는 것을 뼈저리게 느낀 토니는 무기 산업에서 손을 떼며,[7] 주변 사람에게 아크 리액터를 이용한 신에너지 사업에 뛰어들자고 제안하지만 동료인 오베디아나 친구인 로즈한테 젖병 만들어 팔자고? / 지금 네가 해야 할 건 무기를 계속 개발하는 거야는 식의 기운 빠지는 대응을 받게 된다.[8] 이에 상심했는지 홀로 Mk.1을 기반으로 새로운 슈트 제작에 착수한다.[9]
    토니는 Mk.1을 만들었던 경험을 살려, 이번엔 제대로된 설비와 자원을 갖추고 훨씬 세련된 Mk.2를 개발하며, 멋지게 비행에 성공하지만, 호기롭게 고공 비행을 시도하다가 고고도에서 슈트가 결빙하는 문제가 발견되고, 추락해서 죽을 뻔했다가 가까스로 살아나서는 슈트의 소재를 골드 티타늄 합금으로 바꾸어 결빙 문제를 해결한 Mk.3 슈트의 개발에 성공한다. 그러다가 자신의 이름으로 주최되었지만, 정작 자신은 초대받지 못한 파티장에서 다시 만난 크리스틴 에버하트[10]를 통해 은인 호 인센의 고향인 굴미라가 텐 링즈의 횡포에 시달리고 있다는 것을 알게 된 토니는, 끓어오르는 화를 삭이기 위해 처음으로 추진기 시스템을 공격 용도로 테스트해본다. 그런데 전등에 대고 추진기를 쏘아보며 상당한 위력을 확인하고는, 이어 작업실 유리에 3번 정도 더 사격해본 뒤 충분히 공격성능이 있다고 판단한 토니는 즉각 Mk.3를 장착하고 굴미라로 날아가 텐 링즈를 모두 쓸어버린다.[11] 이후 무기 좌표를 확인한 후 이동하던 도중 텐 링즈의 탱크에 공격당하자 스마트 미사일로 한 방에 처리해버리고, 덤으로 이들이 밀수한 스타크 인더스트리의 무기[12]를 모두 박살낸다.
    테러리스트들을 사살하고 무기를 없앤 후 귀환하던 중, 미 공군의 방공레이더에 걸려 F-22기의 추격을 받고, 이 추격 작전을 지휘하던 친우 제임스 로드에게 해당 타겟이 본인임을 밝히고, 로드는 토니에게 나중에 대체 무슨 일을 벌인건지 설명하라고 말을 한다. 이때 로드와 전화 통화를 하느라 임시로 F-22 1대의 아랫부분에 매달려 있었는데, 이를 통신으로 인지한 해당 전투기가 급격한 선회기동을 하면서 전투기를 놓쳐버려 날아가던 와중에 날개와 부딪혀 격추된 다른 전투기의 조종사를[13] 구조해주기도 했다. 그렇게 성공적으로 테러리스트에 대한 공격이 끝난 줄 알았지만, 자신 몰래 테러리스트들에게 회사의 무기를 암거래한 범인이 자신에겐 아버지와 같았던 오베디아 스탠이였음을 알게 되면서 충격을 받는다. 설상가상으로 오베디아는 이사회를 동원해 스타크 인더스트리에서 토니의 지분을 없애나가고 있었으며, 라자에게 입수한 Mk.1의 잔해와 설계도를 기반으로 몰래 아이언 몽거를 제작하고 있었고, 페퍼의 조사를 통해 오베디아가 토니를 죽이려고 텐 링즈에 팔아넘겼다는 사실도 알게 된다.[14] 하지만 페퍼의 연락을 받은 직후, 자신이 들켰음을 깨닫고 선수를 친 오베디아가 아이언 몽거를 완성시키기 위해 토니를 제압하고 아크 리액터를 탈취해 가는 바람에[15] 토니는 죽음의 위기에 봉착한다. 그러나 다행히 페퍼 포츠가 보관해 둔 구형 리액터를 대용품으로 써서 목숨을 건진다.[16] [17]
    이후 Mk.3 슈트를 장착하고 페퍼를 죽이려 드는 아이언 몽거와 싸우지만, 이미 오래된 아크 리액터를 장착한 토니의 슈트의 출력과 에너지가 부족해 밀리게 된다. 상공으로 올라가 결빙현상을 개량하지 않은 아이언 몽거를 추락시키지만 바로 이어서 슈트의 에너지가 다해 본인도 추락하고 만다. 비상동력을 이용해 간신히 자세를 제어하며 감속을 걸어 스타크 인더스트리 옥상에 착륙하지만 아이언 몽거는 아무렇지도 않게 올라와 건물 옥상에서 토니를 밀어붙인다. 하필이면 아이언 몽거가 떨어지면서 파괴된 줄 알고 슈트를 해체하던 도중이었는지라[18] 맨손으로 아이언 몽거에게 맞서며 플레어를 터뜨리고서 사격통제장치의 회로를 뜯어내는 등 어느 정도 싸워보지만 결국 파워에서 밀려버리고 만다. 그나마 사통장치의 회로를 뜯어놔서 당장 미사일에 맞아죽는 상황은 피했지만 그렇다고 딱히 그 상황에서 이길 수 있는 방법이 있는 것도 아닌 마당에, 페퍼의 도움으로 자살이나 다름없는 연구용 아크 리액터 폭파 작전으로 겨우 아이언 몽거를 쓰러뜨리는데 성공한다. 모든 사건이 정리된 후, S.H.I.E.L.D가 준비해 준 대본에 따라 아이언맨의 정체를 감추기로 한다. 그러나 기자회견장에서 얌전히 대본대로 읽으려고 했으나, 원래부터 무언가를 숨기는데 익숙치 않은데다 낌새를 채고 있었던 크리스틴의 날카로운 질문에 계속해서 횡설수설하게 되고, 결국 마지막 순간 대본 따위는 집어치우고, "제가 바로 아이언맨입니다."라고 폭탄 발언을 하면서 영화는 끝이 난다.[19]
    기존의 영웅물과 달리 영웅의 정체성이나 의무에 대한 고뇌는 거의 없는 것이 특징이자 개성. 워낙 마음을 굳세게 먹었고, 독선적일 정도로 자신이 갈 길을 확실히 정했고 망설임이 없었다. 또한 흔한 개과천선형 캐릭터들과는 달리 전이나 지금이나 까불거리는 건 여전하다. 담당 배우 로버트 다우니 주니어는 "아무리 마음을 고쳐먹어도 행동양식까지 다 바뀌는 것은 비현실적이지 않냐"고 말했다. 그래도 손실을 감수하면서 군수산업을 폐기하고 직접 Mk.3타고 날아가서 무기도 파괴하고 인센의 마을을 비롯한 여러 마을을 구했으며, 아이언 몽거와 싸우는 와중에 휘말린 시민들도 구하고, 아이언 몽거도 쓰러뜨리는 등 선행은 확실히 했다. 엠파이어지에서 최고의 영화 캐릭터에 13위로 뽑았으며 여태까지 나온 만화책 원작 캐릭터 중 가장 생동감 있다고 평가했다.
    '''
    db_manager.add_persona(persona, user_id)
    
    # init submodules
    on_llm_new_sentence_handler = lambda sentence: print(f'{user_id}: {sentence}')
    on_llm_end_handler = lambda _: print('Generation Complete.')
    generator = OpenAIGenerator(GENERATOR_MODEL, OPENAI_API_KEY)
    summarizer = Summarizer(SUMMARIZER_MODEL)
    chatbot = Chatbot(generator, summarizer, db_manager)
    
    # generate response
    query = "나는 로드. 토니, 너의 슈트가 처음 만들어진 날을 기억해?"
    _, summary = chatbot.generate(
        query, 
        user_id,
        on_llm_new_sentence_handler=on_llm_new_sentence_handler,
        on_llm_end_handler=on_llm_end_handler
    )
    print(f'Dialogue summary : {summary}')
    
    query = "나는 로드. F22에 쫒기고 있을 때, 내가 구해준거 평생 잊지 않는다고 했잖아."
    _, summary = chatbot.generate(
        query, 
        user_id, 
        f'로드와 토니의 대화. {summary}',
        on_llm_new_sentence_handler=on_llm_new_sentence_handler,
        on_llm_end_handler=on_llm_end_handler
    )
    print(f'Dialogue summary : {summary}')
    
    # close test
    db_manager.delete_personas_by_user_id(user_id)
    print('----- FINISH INTEGRATION TEST -----')

if __name__ == "__main__":
    
    print('----- ENVIRONMENT -----')
    print(f'OPENAI_API_KEY: {OPENAI_API_KEY}')
    print(f'GENERATOR_MODEL: {GENERATOR_MODEL}')
    print(f'EMBEDDING_MODEL: {EMBEDDING_MODEL}')
    print(f'DB_PORT: {DB_PORT}')
    print(f'SUMMARIZER_MODEL: {SUMMARIZER_MODEL}')
    print('------------------------')
    
    test_all_unit()
    test_chatbot()