import json
from collections import Counter
import pickle
import sys
import re
from tqdm import tqdm

kw_or_builtin = set(('abstract continue for new switch assert default goto package synchronized '
     'boolean do if private this break double implements protected throw byte else '
     'import public throws case enum instanceof return transient catch extends int '
     'short try char final interface static void class finally long strictfp '
     'volatile const float native super while true false null').split(' '))

subtok_counter, other_counter = Counter(), Counter()

def is_token(s):
    if not (s[0].isalpha() or s[0] in '$_'):
        return False
    return s not in kw_or_builtin
    
def update_counts(s):
    if not all(ord(c) < 128 for c in s): # only handle ASCII
        return
    if is_token(s):
        subtok_counter.update(subtokenize(s))
    else:
        other_counter.update([s])
    
def subtokenize(token):
    c_style = [st for st in token.split('_') if len(st) > 0]
    subtokens = []
    for subtoken in c_style:
        # from https://stackoverflow.com/questions/29916065/how-to-do-camelcase-split-in-python
        camel_splits = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', subtoken)
        subtokens += [split.group(0).lower() for split in camel_splits]
    return subtokens

fin = open('java_names/output.json', 'r')
file_seeks = [0]
file_size = 71967301632
line_i, last_seek = 0, 0
bar = tqdm(total = file_size, desc = 'file processed', unit = 'bytes')

while True:
    line = fin.readline()
    if len(line) == 0:
        break
    data = json.loads(line)
    cur_seek = fin.tell()
    file_seeks.append(cur_seek)
    update_counts(data['variableName'])
    for context in data['usage']:
        for tok in context:
            update_counts(tok)
            
    bar.update(cur_seek - last_seek)
    last_seek = cur_seek
    line_i += 1
    if line_i % 100000 == 0:
        with open('data/counters_%.9d.pkl' % line_i, 'wb') as counter_file:
            pickle.dump((subtok_counter, other_counter), counter_file)
        with open('data/json_seeks_%.9d.pkl' % line_i, 'wb') as seeks_file:
            pickle.dump(file_seeks, seeks_file)
        
bar.close()
fin.close()

with open('data/counters.pkl', 'wb') as counter_file:
    pickle.dump((subtok_counter, other_counter), counter_file)
with open('data/json_seeks.pkl', 'wb') as seeks_file:
    pickle.dump(file_seeks, seeks_file)
