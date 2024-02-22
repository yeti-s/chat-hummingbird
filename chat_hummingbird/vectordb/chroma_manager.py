import chromadb

from transformers import AutoTokenizer
from chromadb import GetResult
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
        chunk_size:int = 128,
        chunk_overlap:int = 24,
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
        
    def get_persona_by_user_id(self, user_id:str) -> GetResult:
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
    
    def add_message(self, summary:str, relation:str, dialogue:str) -> str:
        doc = Document(
            page_content=summary,
            metadata={
                'relation': relation,
                'dialogue': dialogue
            }
        )
        
        return self.messages_db.add_documents([doc])
    
    def search_message(self, query:str, relation:str) -> str:
        docs = self.messages_db.similarity_search(
            query=query,
            k=1,
            filter={'relation': relation}
        )
        if len(docs) == 0:
            return []
        
        return docs[0].metadata['dialogue']
        
        
    
        
    def set_splitter(self, chunk_size:int, chunk_overlap:int) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function = lambda x: len(self.tokenizer(x)['input_ids']),
            separators = ['\n', '\n\n', '.', '?', '!']
        )

