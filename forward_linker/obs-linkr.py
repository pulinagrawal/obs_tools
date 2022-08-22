# pip3 install pyperclip pyyaml python-frontmatter
from ast import parse
import frontmatter
import pyperclip
import yaml
import os
import re
from pathlib import Path
import pickle as pkl

unlinkr = __import__('obs-unlinkr')

page_aliases = {}
generated_aliases = {}
rev_alias_dict = {}
obsidian_home = ''
wikipedia_mode = False
paragraph_mode = False
yaml_mode = False
regenerate_aliases = False
clear_links = False
dont_use_orphans = False
build_cache = False
vault = False
fuzzy_match = False
r=None
model=None
st = None

def get_similarity(s1, s2):
    return util.pytorch_cos_sim(s1, s2)[0].numpy().item()

def link(title, updated_txt, offset, start, end):
    cont = False
    ret = False
    txt_to_link = updated_txt[start + offset:end + offset]
    
    # where is the next ]]?
    next_closing_index = updated_txt.find("]", end + offset)
    # where is the next [[?
    next_opening_index = updated_txt.find("[", end + offset)   
    # find if it is in a hyperlink
    last_space = updated_txt.rfind(' ', 0, start+offset)
    last_newline = updated_txt.rfind('\n', 0, start+offset)
    last_sep = max(last_space, last_newline)
    if last_sep == -1:
        last_sep = 0
    if len(list( re.finditer("https://|file://|ftp://|www\.|#", updated_txt[last_sep:start+offset]) )):
        cont=True
        return updated_txt, offset, cont, ret 
    # only proceed to link if our text is not already enclosed in a link
    # don't link if there's a ]] ahead, but no [[ (can happen with first few links)
    if not (next_opening_index == -1 and next_closing_index > -1):
        # proceed to link if no [[ or ]] ahead (first link) or [[ appears before ]]
        if (next_opening_index == -1 and next_closing_index == -1) or (next_opening_index < next_closing_index):
            updated_title = title
            # handle aliases
            if title in rev_alias_dict: updated_title = rev_alias_dict[title]
            # handle the display text if it doesn't match the page title
            if txt_to_link != updated_title: updated_title += '|' + txt_to_link
            # create the link and update our text
            updated_txt = updated_txt[:start + offset] + '[[' + updated_title + ']]' + updated_txt[end + offset:]
            # change our offset due to modifications to the document
            offset = offset + (len(updated_title) + 4 - len(txt_to_link))  # pairs of double brackets adds 4 chars
            # if wikipedia mode is on, return after first link is created
            if wikipedia_mode: 
                ret = True
                return updated_txt, offset, cont, ret

    return updated_txt, offset, cont, ret 

def link_title(title, txt):
    updated_txt = txt
    # find instances of the title where it's not surrounded by [], | or other letters
    matches = re.finditer('(?<!([\[\w\|]))' + re.escape(title.lower()) + '(?!([\|\]\w]))', txt.lower())
    offset = 0 # track the offset of our matches (start index) due to document modifications
    
    for m in matches:
        updated_txt, offset, cont, ret = link(title, updated_txt, offset, m.start(), m.end())        # get the original text to link
        if cont:
            continue
        if ret:
            return updated_txt 

    return updated_txt

def update_frontmatter(title, keyword):
    found = False
    if title in rev_alias_dict:
        title = rev_alias_dict[title]
    for root, dirs, files in os.walk(obsidian_home):
        for file in files:
            if file.endswith('.md') and title in file:
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    fm = frontmatter.load(f)
                    if 'aliases' in fm:
                        if keyword not in fm['aliases']:
                            fm['aliases'].append(keyword)
                    else:
                        fm['aliases'] = [keyword]
                post_str = frontmatter.dumps(fm) 
                with open(os.path.join(root, file), 'w', encoding='utf-8') as f:
                    f.write(post_str)
                found = True
                break
        if found:
            break
        else:
            continue
        
    if not found:
        (Path(obsidian_home)/'aliases').mkdir(parents=True, exist_ok=True)
        with open(os.path.join(obsidian_home, 'aliases', title + '.md'), 'w', encoding='utf-8') as f:
            f.write('---\naliases: ["' + keyword + '"]\n---\n')

def fuzzy_match_all_titles(page_titles, content, title_embeddings):
    r.extract_keywords_from_text(content.replace('[[', '').replace(']]', ''))
    ranked_phrases = r.get_ranked_phrases()
    ranked_phrases += [phrase.strip() for phrase in re.split('\.|\?|,|!|:|;|\n', content)]
    for keyword in sorted(ranked_phrases):
        keyword_embedding = model.encode(keyword, convert_to_tensor=True)
        scores = {title: get_similarity(title_emb, keyword_embedding) 
                            for title, title_emb in zip(page_titles, title_embeddings)}
        best_title = max(scores, key=lambda key: scores[key])
        if scores[best_title] > 0.85:
            update_frontmatter(best_title, keyword)
            start = content.find(keyword)
            end = start+len(keyword)
            if start==-1 or '[[' in content[start:end] or ']]' in content[start:end]:
                continue
            updated_txt, _, _, _ = link(best_title, content, 0, start, end)
            if len(updated_txt) != len(content):
                content = updated_txt
                print("linked %s" % keyword)

    return content

def link_content(content, page_titles, title_embeddings=None):
    # make a copy of our content and lowercase it for search purposes
    content_low = content.lower()

    # iterate through our page titles
    for page_title in sorted(page_titles, reverse=True):
        # if we have a case-insenitive title match...
        if page_title.lower() in content_low:        
            updated_txt = link_title(page_title, content)            
            # we can tell whether we've matched the term if
            # the linking process changed the updated text length
            if len(updated_txt) != len(content):
                content = updated_txt
                print("linked %s" % page_title)

        # lowercase our updated text for the next round of search
        content_low = content.lower()        
        
    if fuzzy_match:
        content = fuzzy_match_all_titles(page_titles, content, title_embeddings=title_embeddings)
    
    return content

def build_page_titles():
    page_titles = []
    aliases_file = obsidian_home / ("aliases" + (".yml" if yaml_mode else ".md"))

    # get a directory listing of obsidian *.md files
    # use it to build our list of titles and aliases
    for root, dirs, files in os.walk(obsidian_home):
        for file in files:
            # ignore any 'dot' folders (.trash, .obsidian, etc.)
            if file.endswith('.md') and '\\.' not in root and '/.' not in root:
                page_title = re.sub(r'\.md$', '', file)
                #print(page_title)
                page_titles.append(page_title)
                
                # load yaml frontmatter and parse aliases
                if regenerate_aliases:
                    try:
                        with open(root + "/" + file, encoding="utf-8") as f:
                            #print(file)
                            fm = frontmatter.load(f)
                            
                            if fm and 'aliases' in fm:
                                #print(fm['aliases'])
                                generated_aliases[page_title] = fm['aliases']
                    except yaml.YAMLError as exc:
                        print("Error processing aliases in file: " + file)
                        exit()

    # if -r passed on command line, regenerate aliases.yml
    # this is only necessary if new aliases are present
    if regenerate_aliases:
        with open(aliases_file, "w", encoding="utf-8") as af:
            for title in generated_aliases:
                af.write(title + ":\n" if yaml_mode else "[[" + title + "]]:\n")
                #print(title)
                for alias in generated_aliases[title]:
                    af.write("- " + alias + "\n")
                    #print(alias)
                af.write("\n")
            if not yaml_mode: af.write("aliases:\n- ")

        global rev_alias_dict
        with open(aliases_file, 'r') as stream:
            try:
                # this line injects quotes around wikilinks so that yaml parsing won't fail
                # we remove them later, so they are only a temporary measure
                aliases_txt = stream.read().replace("[[", "\"[[").replace("]]", "]]\"")
                aliases = yaml.full_load(aliases_txt)
                for title in aliases:
                    for alias in aliases[title]:
                        # strip out wikilinks and quotes from title if present
                        sanitized_title = title.replace("[[", "").replace("]]", "").replace("\"", "")
                        if alias:
                            rev_alias_dict[alias] = sanitized_title
            except yaml.YAMLError as exc:
                print(exc)
                exit()


    # load the aliases file
    # we pivot (invert) the dict for lookup purposes
    if os.path.isfile(aliases_file):
        with open(aliases_file, 'r') as stream:
            try:
                # this line injects quotes around wikilinks so that yaml parsing won't fail
                # we remove them later, so they are only a temporary measure
                aliases_txt = stream.read().replace("[[", "\"[[").replace("]]", "]]\"")
                aliases = yaml.full_load(aliases_txt)
                
                if aliases:
                    for title in aliases:         
                        if aliases[title]:                  
                            for alias in aliases[title]:
                                # strip out wikilinks and quotes from title if present
                                sanitized_title = title.replace("[[", "").replace("]]", "").replace("\"", "")
                                if alias:
                                    page_aliases[alias] = sanitized_title
                                else:
                                    # empty entry will signal to ignore page title in matching
                                    print("Empty alias (will be ignored): " + sanitized_title)
                                    if sanitized_title in page_titles:
                                        page_titles.remove(sanitized_title)
                        #print(page_aliases)
            except yaml.YAMLError as exc:
                print(exc)
                exit()

    # append our aliases to the list of titles
    for alias in page_aliases:
        page_titles.append(alias)
    # sort from longest to shortest page titles so that we don't
    # identify scenarios where a page title is a subset of another
    page_titles = sorted(page_titles, key=lambda x: len(x), reverse=True)
    return page_titles

def load_titles_and_embeddings(titles_file, embs_file, load_embs=True):
    page_titles = []
    title_embeddings = None
    if titles_file.exists():
        with titles_file.open('r') as f:
            # remove new line
            links = [line[:-1] for line in f]
        page_titles += links
    if fuzzy_match and embs_file.exists() and load_embs:
        with embs_file.open('rb') as f:
            title_embeddings = pkl.load(f)
    return page_titles, title_embeddings

def load_page_titles():
    # if links cache file exists add all links to page_titles for linking
    page_titles = []
    title_embeddings = None
    if dont_use_orphans:
        link_cache = non_orphan_links
        link_embs_file = non_orphan_link_emb 
    else:
        link_cache = orphan_links
        link_embs_file = orphan_link_emb

    return load_titles_and_embeddings(link_cache, link_embs_file)

def write_if_different(all_links, titles, link_file, embs_file):
    all_links = set(all_links)
    titles = set(titles)
    different = len(all_links) != len(titles)
    if not different:
        for link in all_links:
            if link not in titles:
                different = True
    if different:
        with open(str(link_file), 'w') as f:
            for link in all_links:
                f.write(link+'\n')
        if not model:
            get_model()
        title_embeddings = model.encode([page_title for page_title in all_links], 
                                        convert_to_tensor=True)
        with open(str(embs_file), 'wb') as f:
            pkl.dump(title_embeddings, f)
        
def get_orphans():
    all_links = []
    for root, dirs, files in os.walk(obsidian_home):
        for file in files:
            # ignore any 'dot' folders (.trash, .obsidian, etc.)
            if file.endswith('.md') and '\\.' not in root and '/.' not in root:
                with open(root + "/" + file, encoding="utf-8") as f:
                    for line in f:
                        matches = re.finditer('\[\[(.*?)\]\]', line)
                        links = [line[m.start():m.end()].lower() for m in matches]
                        for ls in links:
                            ls = ls.strip('[[').strip(']]').strip()
                            if '|' in ls:
                                rev_alias_dict[ls.split('|')[1]] = ls.split('|')[0]
                            all_links += [l for l in ls.split('|')]
    return all_links


def build_links():
    print("Building non-orphan links cache...")
    built_links = build_page_titles() 
    stored_titles, _ = load_titles_and_embeddings(non_orphan_links, non_orphan_link_emb, load_embs=False) 
    write_if_different(built_links, stored_titles, non_orphan_links, non_orphan_link_emb)
    print("Non-orphan links cache and embeddings built.")
    print("Building orphan links cache...")
    built_links += get_orphans()
    aliases_file = obsidian_home / ("aliases" + (".yml" if yaml_mode else ".md"))
    # load the aliases file
    # we pivot (invert) the dict for lookup purposes
    if os.path.isfile(aliases_file):
        with open(aliases_file, 'r') as stream:
            try:
                # this line injects quotes around wikilinks so that yaml parsing won't fail
                # we remove them later, so they are only a temporary measure
                aliases_txt = stream.read().replace("[[", "\"[[").replace("]]", "]]\"")
                aliases = yaml.full_load(aliases_txt)
                
                if aliases:
                    # build rev_aliases file
                    for title in aliases:         
                        if aliases[title]:                  
                            for alias in aliases[title]:
                                # strip out wikilinks and quotes from title if present
                                sanitized_title = title.replace("[[", "").replace("]]", "").replace("\"", "")
                                if alias:
                                    page_aliases[alias] = sanitized_title
                                else:
                                    # empty entry will signal to ignore page title in matching
                                    print("Empty alias (will be ignored): " + sanitized_title)
                                    if sanitized_title in built_links:
                                        built_links.remove(sanitized_title)
                        #print(page_aliases)
            except yaml.YAMLError as exc:
                print(exc)
                exit()

    # append our aliases to the list of titles
    for alias in page_aliases:
        built_links.append(alias)

    with open(obsidian_home / "rev_aliases.yml", 'w') as f:
        yaml.dump(rev_alias_dict, f)

    stored_titles, _ = load_titles_and_embeddings(orphan_links, orphan_link_emb, load_embs=False) 
    write_if_different(set(built_links), stored_titles, orphan_links, orphan_link_emb)
    print("Orphan links cache and embeddings built.")

def run_on_text(text, page_titles, title_embeddings):
    # unlink text prior to processing if enabled
    if (clear_links):
        text = unlinkr.unlink_text(text)
        #print('--- text after scrubbing links ---')
        #print(clip_txt)
        #print('----------------------')

    # prepare our linked text output
    linked_txt = ""

    if paragraph_mode:
        for paragraph in text.split("\n"):
            linked_txt += link_content(paragraph, page_titles, title_embeddings=title_embeddings) + "\n"
        linked_txt = linked_txt[:-1] # scrub the last newline
    else:
        linked_txt = link_content(text, page_titles, title_embeddings=title_embeddings)
    return linked_txt

def run_on_clipboard(page_titles, title_embeddings=None):
    # get text from clipboard
    clip_txt = pyperclip.paste()
    #print('--- clipboard text ---')
    #print(clip_txt)
    print('----------------------')

    linked_txt = run_on_text(clip_txt, page_titles, title_embeddings=title_embeddings)

    # send the linked text to the clipboard
    pyperclip.copy(linked_txt)
    #print(clip_txt)
    print('----------------------')
    print('linked text copied to clipboard')

def run_on_file(filepath, page_titles, title_embeddings=None):
    filepath = Path(filepath)
    file = str(filepath.name)
    root = str(filepath.parent)
    with open(root + "/" + file, encoding="utf-8") as f:
        text = f.read()
    linked_text = run_on_text(text, page_titles, title_embeddings=title_embeddings)
    with open(root + "/" + file, 'w', encoding="utf-8") as f:
        f.write(linked_text)

def run_on_vault(page_titles, title_embeddings=None):
    for root, dirs, files in os.walk(obsidian_home):
        for file in files:
            # ignore any 'dot' folders (.trash, .obsidian, etc.)
            if file.endswith('.md') and '\\.' not in root and '/.' not in root:
                run_on_file(root + "/" + file, page_titles, title_embeddings=title_embeddings)


from argparse import ArgumentParser
parse = ArgumentParser(description='Link text in Obsidian Vault')
parse.add_argument('obsidian_home', type=str, help='vault folder location')
parse.add_argument('-r', '--regenerate_aliases', action='store_true', 
    help='regenerate the aliases.md file using yaml frontmatter inside vault markdown files')
parse.add_argument('-w', '--wikipedia_mode', action='store_true', 
    help='only the first occurrence of a page title (or alias) in the content will be linked ("wikipedia mode")')
parse.add_argument('-p', '--paragraph_mode', action='store_true', 
    help='only the first occurrence of a page title (or alias) in each paragraph will be linked ("paragraph mode")')
parse.add_argument('-y', '--yaml_mode', action='store_true', 
    help='use aliases.yml as aliases file instead of aliases.md')
parse.add_argument('-u', '--clear_links', action='store_true', 
    help='remove existing links in clipboard text before performing linking')
parse.add_argument('-x', '--dont_use_orphans', action='store_true', 
    help='dont use orphan links when linking')
parse.add_argument('-b', '--build_cache', action='store_true', 
    help='build all links cache file and exit (ignores all other flags)')
parse.add_argument('-z', '--fuzzy_match', action='store_true', 
    help='use fuzzy semantic similarity match when linking')
group = parse.add_mutually_exclusive_group(required=False)
group.add_argument('-v', '--vault', action='store_true', 
    help='run linker on the whole vault')
group.add_argument('-c', '--clipboard', action='store_true', 
    help='run linker on the content in clipboard')
group.add_argument('-f', '--file', type=str, 
    help='run linker on the file specified')
args = parse.parse_args()


for arg in vars(args):
    if arg in dir():
        globals()[arg] = getattr(args, arg)

obsidian_home = Path(obsidian_home)
non_orphan_links = obsidian_home/'.obsidian'/'non_orphan_links.cache'
non_orphan_link_emb = obsidian_home/'.obsidian'/'non_orphan_link_emb.cache'
orphan_links = obsidian_home/'.obsidian'/'orphan_links.cache'
orphan_link_emb = obsidian_home/'.obsidian'/'orphan_link_emb.cache'

def get_model():
    global model, st, util
    if not st:
        from sentence_transformers import SentenceTransformer as st, util
    model = st('stsb-roberta-large')
    
if build_cache:
    get_model()
    build_links()
    print('Rebuilt orphans links cache file.')

    
if args.file or args.vault or args.clipboard:
    if args.fuzzy_match:
        from rake_nltk import Rake
        import nltk
        nltk.download('stopwords');
        nltk.download('punkt');
        r = Rake(include_repeated_phrases=False)
        get_model()

    with open(obsidian_home/'rev_aliases.yml', 'r') as f:
        rev_alias_dict = yaml.safe_load(f)

    print('Collecting links...')
    page_titles, title_embeddings = load_page_titles()

    print('Running linker...')
    if args.file:
        run_on_file(filepath=args.file, page_titles=page_titles, title_embeddings=title_embeddings)
    elif args.vault_run:
        run_on_vault(page_titles, title_embeddings=title_embeddings)
    else:
        run_on_clipboard(page_titles, title_embeddings=title_embeddings)

    print('Linker ran.')