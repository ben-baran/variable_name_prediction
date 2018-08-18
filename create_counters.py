import json
from collections import Counter
import pickle
import sys
from tqdm import tqdm
from utils import is_token, subtokenize

subtok_counter, other_counter = Counter(), Counter()
    
def update_counts(s):
    if not all(ord(c) < 128 for c in s): # only handle ASCII
        return
    if is_token(s):
        subtok_counter.update(subtokenize(s))
    else:
        other_counter.update([s])

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
