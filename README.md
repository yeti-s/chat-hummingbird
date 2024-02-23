# Chatbot 💬

여러 LLM을 이용하여 한국어로 사용자와 페르소나 기반 대화를 할 수 있는 챗봇을 만듭니다.

# Install 📲

### develop

```python
$ pip install .
```

### package
```
$ pip install git+https://github.com/yeti-s/chat-hummingbird.git
```

# Update 🆕

### 2024.02.19

* 생성되는 말투의 일관성을 유지하기 위해 답변 생성시 이전까지 나눴던 대화 이력 정보를 함께 넣어주었습니다.
* 짧은 문장으로 이루어진 데이터셋으로 학습된 Embedding 모델의 특성을 반영하여 Vector DB에 페르소나 저장 시 짧은 문장으로 분리하여 저장하도록 변경하였습니다. 

### 2024.02.14

* src, test 코드를 나누도록 폴더 구조를 변경하였습니다.
* pytest를 이용한 unittest를 코드를 추가하였습니다.

### 2024.02.13

* OpenAI API를 이용하여 응답을 생성하는 Generator를 추가하였습니다.
* ChromaDB를 이용하여 사용자의 페르소나에서 응답에 필요한 정보를 검색하는 Retriever를 추가하였습니다.
* HuggingFace의 모델을 활용하여 대화 내용을 요약하는 Summarizer를 추가하였습니다.
* 위 세 모델을 활용하여 적절한 응답을 만들어내는 Chatbot을 추가하였습니다.

# How to use 🤷

최상위 폴더에 .env 파일을 만들고 필요한 설정을 입력해주세요.
```
#.env
# Generator
OPENAI_API_KEY = <DUMMY>
GENERATOR_MODEL = <DUMMY>

# DB Manager
EMBEDDING_MODEL = <DUMMY>
DB_HOST = <DUMMY>
DB_PORT = <DUMMY>

# Summarizer
SUMMARIZER_MODEL = <DUMMY>
```

### Unit Test

pytest 라이브러리를 활용하여 unit test를 진행합니다.

```
$ python -m pytest tests
```

### Manually

chatbot, openai, chroma, summarizer 네 가지 모듈에 대해 테스트를 실행하고 진행 과정을 콘솔에 표시합니다.

```
$ python tests/run_manually.py --summarizer
```
```
----- ENVIRONMENT -----
DB_HOST: <DUMMY>
OPENAI_API_KEY: <DUMMY>
GENERATOR_MODEL: <DUMMY>
EMBEDDING_MODEL: <DUMMY>
DB_PORT: <DUMMY>
SUMMARIZER_MODEL: <DUMMY>
------------------------
====== TEST SUMMARIZATION =====
dialogues : ['오늘 뭐해?', '그냥 공부하고 있지.', '서울대입구에서 술 고?', '7시쯤 가능함!', '7시 오케이!']
오늘 서울대 입구에서 7시쯤 술을 마시기로 했다.
====== FINISH SUMMARIZATION =====
```

# Citation 📜

```
@inproceedings{lee2023kullm,
  title={KULLM: Learning to Construct Korean Instruction-following Large Language Models},
  author={Lee, SeungJun and Lee, Taemin and Lee, Jeongwoo and Jang, Yoona and Lim, Heuiseok},
  booktitle={Annual Conference on Human and Language Technology},
  pages={196--202},
  year={2023},
  organization={Human and Language Technology}
}
```

```
@misc{kullm,
  author = {NLP & AI Lab and Human-Inspired AI research},
  title = {KULLM: Korea University Large Language Model Project},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/nlpai-lab/kullm}},
}
```