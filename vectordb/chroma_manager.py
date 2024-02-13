# -- coding: utf-8 --
import subprocess
import chromadb

from transformers import AutoTokenizer

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

MESSAGE_COLLECTION = 'messages'
PERSONA_COLLECTION = 'persona'

class ChromaManager():
    def __init__(
        self, 
        host:str, 
        port:str, 
        model_name:str,
        chunk_size:int = 256,
        chunk_overlap:int = 0,
    ) -> None:
        self.connected = False
        self.__connect_db__(host, port, model_name)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.set_splitter(chunk_size, chunk_overlap)
    
    def __connect_db__(self, host, port, model_name):
        # connect to DB
        try:
            self.client = chromadb.HttpClient(host=host, port=port)
            print(f'chromadb [version : {self.client.get_version()}] connected.')
            self.connected = True
        except Exception as e:
            print(e)
            return
        
        # get collection
        self.messages = self.__get_collection__(MESSAGE_COLLECTION)
        self.personas = self.__get_collection__(PERSONA_COLLECTION)
        
        embedding_function = SentenceTransformerEmbeddings(
            model_name = model_name,
            model_kwargs = {'device': 'cuda'},
            encode_kwargs= {'normalize_embeddings': True}
        )
        
        self.messages_db = Chroma(
            client = self.client,
            collection_name = MESSAGE_COLLECTION,
            embedding_function = embedding_function
        )
        
        self.personas_db = Chroma(
            client = self.client,
            collection_name = PERSONA_COLLECTION,
            embedding_function = embedding_function
        )
    
    
    def __get_collection__(self, name:str):
        if any(collection.name == name for collection in self.client.list_collections()):
            return self.client.get_collection(name)
        else:
            return self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
        
    
    def add_persona(self, persona:str, user_id:str) -> list[str]:
        doc = Document(page_content=persona, metadata={'user_id': user_id})
        docs = self.splitter.split_documents([doc])
        for i in range(len(docs)):
            docs[i].metadata['index'] = i
        return self.personas_db.add_documents(docs)
    
    def delete_persoan_by_id(self, id:str) -> None:
        self.personas.delete([id])
    
    def delete_personas_by_user_id(self, user_id:str) -> None:
        self.personas.delete(where={'user_id': user_id})
        
    def get_persona_by_user_id(self, user_id:str):
        return self.personas.get(where={'user_id': user_id})

    def search_persona(self, query:str, user_id:str) -> str:
        persona = ''
        docs = self.personas_db.similarity_search(
            query=query,
            k=1,
            filter={'user_id': user_id}
        )
        if len(docs) == 0:
            return persona
        
        index = docs[0].metadata['index']
        docs = self.personas_db.get(where={
            '$and': [
                {
                    'user_id': user_id
                },
                {
                    'index': {'$gte': index-1 if index > 0 else 0}
                },
                {
                    'index': {'$lte': index+1}
                }
        ]})
        persona = ''
        for doc in docs['documents']:
            persona = f'{persona} {doc}'
        return persona
    
        
    def set_splitter(self, chunk_size:int, chunk_overlap:int) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function = lambda x: len(self.tokenizer(x)['input_ids']),
            separators = ['\n', '\n\n', '.', '?', '!']
        )


chroma_process = None

# def run_chroma(host, port):
#     print('===== RUN CHROMA DB')
#     chroma_process = subprocess.Popen(["chroma", "run", "--host", host, "--port", port])
#     time.sleep(1000)

# def stop_chroma():
#     print('===== CLOSE CHROMA DB')
#     if chroma_process is not None:
#         chroma_process.terminate()
#         chroma_process.wait()

def test_chroma_manager(host, port, model_name):
    # run_chroma(host, port)
    
    print('===== CONNECT TO CHROMA DB')
    chroma_manager = ChromaManager(host, port, model_name, chunk_size=15)
    
    print('===== ADD PERSONA TO CHROMA DB')
    persona = '이것은 테스트 페르소나 입니다. Splitter로 자르기 위해 길게 표현했어요. 또 chunk_size도 작게 조절했죠. 테스트 페르소나 였습니다. 감사합니다.'
    user_id = 'TEST-ID-0001'
    print(f'Add [user_id: {user_id}] [persona: {persona}]')
    print(chroma_manager.add_persona(persona, user_id))
    
    print('===== GET PERSONA BY USER ID')
    print(f'Get [user_id: {user_id}]')
    print(chroma_manager.get_persona_by_user_id(user_id))
    
    print('===== SEARCH PERSONA BY USER ID')
    query = '길게 표현'
    print(f'Get [query: {query}], [user_id: {user_id}]')
    print(chroma_manager.search_persona(query, user_id))
    
    print('===== DELETE PERSONA BY USER ID')
    print(f'Delete [user_id: {user_id}]')
    chroma_manager.delete_personas_by_user_id(user_id)
    print(chroma_manager.get_persona_by_user_id(user_id))
    
    # stop_chroma()
    