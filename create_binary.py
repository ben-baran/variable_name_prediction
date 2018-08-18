import random
import pickle
import struct
import json
from tqdm import tqdm
import re
from utils import ComboVocab, to_subtokenized_list


n_ctx = 64 # size of context per side. This is decided earlier in the processing pipeline.
combo_vocab = ComboVocab()

with open('data/json_seeks.pkl', 'rb') as seeks_file:
    file_seeks = pickle.load(seeks_file)

fin = open('java_names/output.json', 'r')

train_save_folder = 'data/train_tmp/'
val_save_folder = 'data/val_tmp/'
test_save_folder = 'data/test_tmp/'
train_context_files, val_context_files, test_context_files = {}, {}, {}
random.shuffle(file_seeks)

max_uses = 5000
proportion_validation = 0.1
proportion_test = 0.1

bar = tqdm(total = min(max_uses, len(file_seeks)), desc = 'processing', unit = 'ctx groups')
for seek_i, seek in enumerate(file_seeks):
    # For every set of contexts this outputs: N_CTX ID BEFORE_CTX AFTER_CTX BEFORE_CTX AFTER_CTX
    fin.seek(seek)
    line = fin.readline()
    if len(line) == 0:
        continue
    data = json.loads(line)
    n_contexts = len(data['usage'])
    
    # Now specify the file for n_contexts. Create a new one if it doesn't exist.
    # Randomly choose from train, validation, or test
    prop = random.random()
    if prop < proportion_validation:
        save_folder = val_save_folder
        context_files = val_context_files
    elif prop < proportion_validation + proportion_test:
        save_folder = test_save_folder
        context_files = test_context_files
    else:
        save_folder = train_save_folder
        context_files = train_context_files
    if n_contexts not in context_files:
        context_files[n_contexts] = open(save_folder + '%d.bin' % n_contexts, 'wb')
    fout = context_files[n_contexts]
    
    variable_subtokens = combo_vocab.to_ids([data['variableName']])[:8] # maxes out number of subtokens to 8
    variable_subtokens.extend(combo_vocab.to_ids(['<<PAD>>' for i in range(8 - len(variable_subtokens))]))
    fout.write(struct.pack('<8I', *variable_subtokens))
    
    for context in data['usage']:
        context_a = to_subtokenized_list(context[:64])[-64:]
        context_b = to_subtokenized_list(context[129:64:-1])[-64:]
        fout.write(struct.pack('<64I', *combo_vocab.to_ids(context_a)))
        fout.write(struct.pack('<64I', *combo_vocab.to_ids(context_b)))
        # print("-" * 100)
        # print(context)
        # for o_token, t_token in zip(context_a, combo_vocab.to_tokens(combo_vocab.to_ids(context_a))):
        #     print(o_token, '-->', t_token)
        # break
    bar.update(1)
    
    if seek_i == max_uses - 1:
        break

bar.close()
fin.close()

for fout in context_files.values():
    fout.close()

# idea: use first, second, third moments of the distributions of weights?
