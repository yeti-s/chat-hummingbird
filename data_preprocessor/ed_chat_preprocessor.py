import os
import json
import zipfile
import shutil
import argparse
import jsonlines

from glob import glob
from dotenv import load_dotenv
from chat_hummingbird.summarizer.summarizer import Summarizer

load_dotenv()

SUMMARIZER_MODEL = os.getenv('SUMMARIZER_MODEL')
EMOTIONS = ['기쁨', '당황', '분노', '불안', '상처', '슬픔']
RELATIONS = ['부모', '부부', '연인', '지인', '직장', '지인', '형제', '친구']

class EmpatheticDialogue():
    def __init__(self, json_data, relation, summarizer) -> None:
        self.relation = relation
        self.summarizer = summarizer
        self.utterances = json_data['utterances']
        self.num_index = len(self.utterances)
        self.cur_index = 0
            
    def next(self, num=6):
        if self.cur_index + num >= self.num_index:
            return None
        # create dialouge from utterances
        dialogue = []
        for i in range(self.cur_index, min(self.num_index, self.cur_index + num)):
            dialogue.append(self.utterances[i]['text'])
        self.cur_index += 2
        # summarize dialogue
        summary = self.summarizer.summarize(dialogue)
        
        return {
            'summary': summary,
            'dialogue': dialogue,
            'relation': self.relation
        }
        


def unzip(zip_file, dest):
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(dest)
    
def unzip_all(root_dir):
    new_dirs = set()
    zip_files = glob(os.path.join(root_dir, '**', '*.zip'), recursive=True)
    for file in zip_files:
        file_dir = os.path.dirname(file)
        file_name = os.path.basename(file)
        
        new_dir = os.path.join(file_dir, file_name.split('.')[0])
        os.makedirs(new_dir, exist_ok=True)
        new_dirs.add(new_dir)

        unzip(file, new_dir)
        
    return new_dirs

# read multi jsons and export [relation].jsonl
def compile_to_jsonl(root_dir, export_dir):
    classfied_data = {}
    for relation in RELATIONS:
        classfied_data[relation] = {'files': []}
        for emotion in EMOTIONS:
            classfied_data[relation][emotion] = 0
            
    def get_emotion_type(dir_name):
        for emotion in EMOTIONS:
            if emotion in dir_name:
                return emotion
    
    def get_relation_type(dir_name):
        for relation in RELATIONS:
            if relation in dir_name:
                return relation
    
    for root, dirs, files in os.walk(root_dir):
        for name in dirs:
            dir_path = os.path.join(root, name)
            emotion = get_emotion_type(dir_path)
            relation = get_relation_type(dir_path)
            if emotion is None or relation is None:
                continue
            
            files = glob(os.path.join(dir_path, '*.json'))
            classfied_data[relation]['files'] = files
            classfied_data[relation][emotion] += len(files)
    
    for relation, item in classfied_data.items():
        os.makedirs(export_dir, exist_ok=True)
        exported = os.path.join(export_dir, f'{relation}.jsonl')
        
        with open(exported, 'w', encoding='utf-8') as f_out:
            num_jsons = 0
            for file in item['files']:
                with open(file, 'r', encoding='utf-8') as f_in:
                    json.dump(json.load(f_in), f_out, ensure_ascii=False)
                    f_out.write('\n')
                num_jsons += 1
                
            print(f'export {num_jsons} data on {exported}') 

def preprocess_raw_data(root_dir, export_dir):
    new_dirs = unzip_all(root_dir)
    
    train_dest = os.path.join(export_dir, 'train')
    val_dest = os.path.join(export_dir, 'val')
    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(val_dest, exist_ok=True)
    
    compile_to_jsonl(os.path.join(root_dir, '01-1.정식개방데이터', 'Training'), train_dest)
    compile_to_jsonl(os.path.join(root_dir, '01-1.정식개방데이터', 'Validation'), val_dest)
    
    for new_dir in new_dirs:
        if os.path.exists(new_dir):
            shutil.rmtree(new_dir)        

def create_eds(jsonl_file, summarizer):
    eds = []
    # get relation by file name
    relation = os.path.basename(jsonl_file)    
    # create Empathetic Dialouges
    with jsonlines.open(jsonl_file) as data:
        for json_data in data:
            eds.append(EmpatheticDialogue(json_data, relation, summarizer))

    return eds

def export_eds(eds, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    with open(dest, 'w', encoding='utf-8') as f_out:
        num_jsons = 0
        for ed in eds:
            dialouge = ed.next()
            while dialouge is not None:
                json.dump(dialouge, f_out, ensure_ascii=False)
                f_out.write('\n')
                num_jsons += 1
                
                dialouge = ed.next()

def main(args):
    data_dir = args.data
    dest_dir = args.dest
    train_dir = os.path.join(dest_dir, 'train')
    val_dir = os.path.join(dest_dir, 'val')
    
    preprocess_raw_data(data_dir, dest_dir)
    
    # create train dataset
    eds = []
    summarizer = Summarizer(SUMMARIZER_MODEL)
    for jsonl_file in glob(os.path.join(train_dir, '*.jsonl')):
        print(f'read {jsonl_file}...')
        eds.extend(create_eds(jsonl_file, summarizer))
    export_eds(eds, os.path.join(dest_dir, 'train.jsonl'))
    
    # create val dataset    
    eds = []
    summarizer = Summarizer(SUMMARIZER_MODEL)
    for jsonl_file in glob(os.path.join(val_dir, '*.jsonl')):
        print(f'read {jsonl_file}...')
        eds.extend(create_eds(jsonl_file, summarizer))
    export_eds(eds, os.path.join(dest_dir, 'val.jsonl'))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='--data {data} --dest {destination}')
    parser.add_argument('--data', required=True, help='root directory of aihub empathetic dialogues')
    parser.add_argument('--dest', required=True, help='destination directory')
    main(parser.parse_args())
    