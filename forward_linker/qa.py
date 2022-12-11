import torch
from tqdm import tqdm
from transformers import pipeline
import os
from pathlib import Path
from transformers import AutoModelWithLMHead, AutoTokenizer

def abstractive_answer(context, question):
    tokenizer = AutoTokenizer.from_pretrained("tuner007/t5_abs_qa")
    model = AutoModelWithLMHead.from_pretrained("tuner007/t5_abs_qa")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    input_text = "context: %s <question for context: %s </s>" % (context,question)
    features = tokenizer([input_text], return_tensors='pt')
    out = model.generate(input_ids=features['input_ids'].to(device), attention_mask=features['attention_mask'].to(device))
    return tokenizer.decode(out[0])

def get_context(obsidian_home):
    """
    It walks through the vault, and for each file, it runs the `run_on_file` function
    
    :param page_titles: a list of all the page titles in your vault
    :param title_embeddings: a dictionary of page titles to embeddings. If you have a pre-trained model,
    you can pass it in here
    """
    context = ''
    for root, dirs, files in os.walk(obsidian_home):
        for file in files:
            # ignore any 'dot' folders (.trash, .obsidian, etc.)
            if file.endswith('.md') and '\\.' not in root and '/.' not in root:
                with (Path(root)/file).open(encoding='utf-8') as f:
                    context += '\n'.join(f.readlines())
    return context

def filewise_abstractive(obsidian_home, question):
    """
    It walks through the vault, and for each file, it runs the `run_on_file` function
    
    :param page_titles: a list of all the page titles in your vault
    :param title_embeddings: a dictionary of page titles to embeddings. If you have a pre-trained model,
    you can pass it in here
    """
    context = ''
    answers = []
    for root, dirs, files in tqdm(os.walk(obsidian_home)):
        for file in tqdm(files):
            # ignore any 'dot' folders (.trash, .obsidian, etc.)
            if file.endswith('.md') and '\\.' not in root and '/.' not in root:
                with (Path(root)/file).open(encoding='utf-8') as f:
                    context = '\n'.join(f.readlines())
                answers.append(abstractive_answer(context, question))
    answers = list(filter(lambda ans: not ans.startswith('<pad> No answer'), answers))
    return answers

def extractive_answer(context, question):
    qa = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')
    answer = qa(context=context, question=question)
    return answer['answer'] 

if __name__ == '__main__':
    from argparse import ArgumentParser
    # parse = ArgumentParser(description='Answer a question about text')
    # parse.add_argument('obsidian_home', type=str, help='vault folder location')
    # parse.add_argument('-q', '--question', type=str, help='The Question')
    # args = parse.parse_args()
    obsidian_home='C:\\Users\\pulin\\Desktop\\Obsidian-Vault' 
    context = get_context(obsidian_home=obsidian_home)
    print(extractive_answer(context=context, question='What is Seyed\'s age?'))
    # context = get_context(obsidian_home=args.obsidian_home)
    # print(extractive_answer(context=context, question=args.question))