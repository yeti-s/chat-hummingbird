# Chatbot 💬

LLM을 이용하여 한국어로 사용자와 다양한 의사소통을 할 수 있는 챗봇을 만듭니다.

# Update 🆕

### 2024.02.13

* OpenAI API를 이용하여 응답을 생성하는 Generator를 추가하였습니다.
* ChromaDB를 이용하여 사용자의 페르소나에서 응답에 필요한 정보를 검색하는 Retriever를 추가하였습니다.
* HuggingFace의 모델을 활용하여 대화 내용을 요약하는 Summarizer를 추가하였습니다.
* 위 세 모델을 활용하여 적절한 응답을 만들어내는 Chatbot을 추가하였습니다.

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