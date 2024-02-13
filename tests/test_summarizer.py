from src.summarizer.summarizer import Summarizer

def run_summarizer(model_name):
    print('====== TEST SUMMARIZATION =====')
    
    summarizer = Summarizer(model_name)
    
    dialogues = ['오늘 뭐해?', '그냥 공부하고 있지.', '서울대입구에서 술 고?', '7시쯤 가능함!', '7시 오케이!']
    print('dialogues :', dialogues)
    summary = summarizer.summarize(dialogues)
    assert type(summary) == str
    print(summary)
    
    print('====== FINISH SUMMARIZATION =====')
    
def test_summarizer(summarizer_model):
    run_summarizer(summarizer_model)