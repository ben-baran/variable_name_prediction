import random
import pickle
import struct
import json
from tqdm import tqdm
import re
from utils import ComboVocab, to_subtokenized_list


n_ctx = 64 # size of context per side
combo_vocab = ComboVocab()

with open('data/json_seeks.pkl', 'rb') as seeks_file:
    file_seeks = pickle.load(seeks_file)

fin = open('java_names/output.json', 'r')

save_folder = 'data/test/'
context_files = {}
random.shuffle(file_seeks)

max_uses = 1000
bar = tqdm(total = min(max_uses, len(file_seeks)), desc = 'processing', unit = 'ctx groups')
for seek_i, seek in enumerate(file_seeks):
    # For every set of contexts this outputs: N_CTX ID BEFORE_CTX AFTER_CTX BEFORE_CTX AFTER_CTX
    fin.seek(seek)
    line = fin.readline()
    if len(line) == 0:
        continue
    data = json.loads(line)
    n_contexts = len(data['usage'])
    
    # Now specify the file for n_contexts. Create a new one if it doesn't exist
    if n_contexts not in context_files:
        context_files[n_contexts] = open(save_folder + '%d.bin' % n_contexts, 'wb')
    fout = context_files[n_contexts]
    
    fout.write(struct.pack('<2I', n_contexts, combo_vocab.to_ids([data['variableName']])[0])) # CORRECT THIS
    for context in data['usage']:
        context_a = to_subtokenized_list(context[:64])[-64:]
        context_b = to_subtokenized_list(context[129:64:-1])[-64:]
        fout.write(struct.pack('<64I', *combo_vocab.to_ids(context_a)))
        fout.write(struct.pack('<64I', *combo_vocab.to_ids(context_b)))
        if "<<PAD>>" in context:
            print("-" * 100)
            print(context)
            for o_token, t_token in zip(context_a[:64], combo_vocab.to_tokens(combo_vocab.to_ids(context_a[:64]))):
                print(o_token, '-->', t_token)
            break
    bar.update(1)
    
    if seek_i == max_uses - 1:
        break

bar.close()
fin.close()

for fout in context_files.values():
    fout.close()

# idea: use first, second, third moments of the distributions of weights?
