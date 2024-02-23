from chat_hummingbird.summarizer.summarizer import Summarizer

def run_summarizer(model_name):
    print('====== TEST SUMMARIZATION =====')
    
    summarizer = Summarizer(model_name)
    
    dialogues = ['성근과 준섭의 대화, 올해 수능이 어려웠다고 말하고 있다.', '성근이 말한다. 성적이면 생각보다 높은 대학에 지원해도 될 것 같다고 한다.']
    print('dialogues :', dialogues)
    summary = summarizer.summarize(dialogues)
    assert type(summary) == str
    print(summary)
    
    print('====== FINISH SUMMARIZATION =====')
    
def test_summarizer(summarizer_model):
    run_summarizer(summarizer_model)