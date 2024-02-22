import os
import argparse
import jsonlines

from tqdm import tqdm
from dotenv import load_dotenv
from chat_hummingbird.vectordb.chroma_manager import ChromaManager

load_dotenv()

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')

def read_jsonl(jsonl_path):
    eds = []
    with jsonlines.open(jsonl_path) as data:
        for json_data in data:
            eds.append(json_data)
            
    return eds


def main(args):
    data_path = args.data
    # data_path = 'data/ED/train.jsonl'
    eds = read_jsonl(data_path)
    
    db_manager = ChromaManager(DB_HOST, DB_PORT, EMBEDDING_MODEL, 512)
    for i in tqdm(range(len(eds))):
        ed = eds[i]
        dialouge = ''
        for i in range(len(ed['dialogue'])):
            if i % 2 == 0:
                dialouge = f"{dialouge}A: {ed['dialogue'][i]}\n"
            else:
                dialouge = f"{dialouge}B: {ed['dialogue'][i]}\n"
        db_manager.add_message(ed['summary'], ed['relation'], dialouge)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='--data {data} --dest {destination}')
    parser.add_argument('--data', required=True, help='jsonl format emphatetic dialouge data')
    main(parser.parse_args())
    # main(1)