import spacy
nlp = spacy.load('en_core_web_sm')
filepath ="C:\\Users\\pulin\\Desktop\\Obsidian-Vault\\Ideas\\Explorations\\AGI Exploration.md"
with open(filepath, 'r') as f:
    text = f.read()
import nltk
for lines in text.split('\n'):
    for sentence in lines.split('.'):
        nltk_pos_tagged=nltk.pos_tag(sentence.split())
        chunk_tree=ntc.parse(nltk_pos_tagged)
        print(chunk_tree)
doc = nlp(text.replace('[[', '').replace(']]', ''))
phrases = set() 
for nc in doc.noun_chunks:
    phrases.add(nc.text)
    phrases.add(doc[nc.root.left_edge.i:nc.root.right_edge.i+1].text)
for phrase in phrases:
    print(phrase)