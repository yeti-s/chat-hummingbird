from src.vectordb.chroma_manager import ChromaManager

def run_chroma_manager(host, port, model_name):
    print('===== TEST CHROMA MANAGER =====')
    print('--- CONNECT TO CHROMA DB')
    chroma_manager = ChromaManager(host, port, model_name, chunk_size=15)
    
    print('--- ADD PERSONA TO CHROMA DB')
    persona = '이것은 테스트 페르소나 입니다. Splitter로 자르기 위해 길게 표현했어요. 또 chunk_size도 작게 조절했죠. 테스트 페르소나 였습니다. 감사합니다.'
    user_id = 'TEST-ID-0001'
    print(f'Add [user_id: {user_id}] [persona: {persona}]')
    result = chroma_manager.add_persona(persona, user_id)
    assert len(result) > 0
    print(result)
    
    print('--- GET PERSONA BY USER ID')
    print(f'Get [user_id: {user_id}]')
    result = chroma_manager.get_persona_by_user_id(user_id)
    assert len(result['documents']) > 0
    print(result)
    
    print('--- SEARCH PERSONA BY USER ID')
    query = '길게 표현'
    print(f'Get [query: {query}], [user_id: {user_id}]')
    result = chroma_manager.search_persona(query, user_id)
    assert len(result) > 0
    print(result)
    
    print('--- DELETE PERSONA BY USER ID')
    print(f'Delete [user_id: {user_id}]')
    chroma_manager.delete_personas_by_user_id(user_id)
    reulst = chroma_manager.get_persona_by_user_id(user_id)
    assert len(reulst['documents']) == 0
    print(reulst)
    print('===== FINISH CHROMA MANAGER =====')

def test_chroma_manager(db_host, db_port, embedding_model):
    run_chroma_manager(db_host, db_port, embedding_model)