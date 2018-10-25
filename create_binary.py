import random
import pickle
import struct
import json
from tqdm import tqdm
import re
from utils import ComboVocab, to_subtokenized_list, subtokenize, rough_print

n_ctx = 64 # size of context per side. This is decided earlier in the processing pipeline.
combo_vocab = ComboVocab()

with open('data/json_seeks.pkl', 'rb') as seeks_file:
    file_seeks = pickle.load(seeks_file)

with open('data_options.json') as options_f:
    options = json.load(options_f)
    
fin = open('java_names/output.json', 'r')

train_save_folder = 'data/train_full/'
val_save_folder = 'data/val_full/'
test_save_folder = 'data/test_full/'
max_uses = int(1e32)
max_pred_subtokens = options['max_subtokens_predicted']
proportion_validation = options['proportion_validation']
proportion_test = options['proportion_test']
ctx_width = options['context_width']

train_context_files, val_context_files, test_context_files = {}, {}, {}
random.shuffle(file_seeks)
n_unclear_skips = 0
MAX_OPEN_FILES_PER = 170


n_processed = 0
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
    
    
    var_subtokens = subtokenize(data['variableName'])
    var_subids = combo_vocab.to_ids(var_subtokens)[:max_pred_subtokens]
    
    unclear_subtoken = False
    for subtoken in combo_vocab.to_tokens(var_subids): # turn it back into a token to check, so that we can skip over unclear subtokens
        if subtoken == '<unk>':
            unclear_subtoken = True
            n_unclear_skips += 1
            break
    if not unclear_subtoken:
        var_subids.extend(combo_vocab.to_ids(['<<PAD>>' for i in range(max_pred_subtokens - len(var_subids))]))
        fout.write(struct.pack('<%dI' % max_pred_subtokens, *var_subids))
        for context in data['usage']:
            context_a = to_subtokenized_list(context[:ctx_width])[-ctx_width:]
            context_b = to_subtokenized_list(context[ctx_width+1:])[ctx_width-1::-1]
            fout.write(struct.pack('<%dI' % ctx_width, *combo_vocab.to_ids(context_a)))
            fout.write(struct.pack('<%dI' % ctx_width, *combo_vocab.to_ids(context_b)))
    
    if len(context_files) > MAX_OPEN_FILES_PER:
        highest_open = max(context_files.keys())
        context_files[highest_open].close()
        del context_files[highest_open]
    
    n_processed += 1
    bar.update(1)
    if n_processed == max_uses:
        break

bar.close()
fin.close()

for fout in context_files.values():
    fout.close()

print('Number of context groups skipped due to uncommon prediction:', n_unclear_skips)
    
# idea: use first, second, third moments of the distributions of weights?
