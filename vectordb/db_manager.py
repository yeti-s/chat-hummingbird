from abc import ABCMeta, abstractmethod

class DBManager(metaclass=ABCMeta):
    @abstractmethod
    def add_persona(self, persona:str, user_id:str) -> list[str]:
        pass
    @abstractmethod
    def delete_persoan_by_id(self, id:str) -> None:
        pass
    @abstractmethod
    def delete_personas_by_user_id(self, user_id:str) -> None:
        pass
    @abstractmethod
    def search_persona(self, user_id:str, index:int) -> str:
        pass