import random
import pickle
import struct
import json
import mxnet.contrib.text as mtext
from tqdm import tqdm
import re
from utils import to_subtokenized_list

min_for_vocabulary = 4096
n_ctx = 64 # size of context per side

with open('data/counters.pkl', 'rb') as counter_file:
    subtok_counter, other_counter = pickle.load(counter_file)
with open('data/json_seeks.pkl', 'rb') as seeks_file:
    file_seeks = pickle.load(seeks_file)

subtok_counter['_'] = 4096
subtok_counter['__'] = 4096 # I had forgotten that these can exist on their own
subtok_vocab = mtext.vocab.Vocabulary(subtok_counter, min_freq = min_for_vocabulary)
n_subtoks = len(subtok_vocab.token_to_idx) - 1 # remove one for unknowns
n_others = len(subtok_vocab.token_to_idx) # unknown remains
other_vocab = mtext.vocab.Vocabulary(other_counter, min_freq = min_for_vocabulary)

print("Number of subtokens: %d, number of other tokens: %d" % (n_subtoks, len(other_vocab.token_to_idx) - 1))

literal_constants = ['char', 'string', 'float', 'double', 'hex_int', 'bin_int', 'int']
const_to_literal = {s:i for i, s in enumerate(literal_constants)}
def to_ids(l):
    indices = subtok_vocab.to_indices(l)
    for i, index in enumerate(indices):
        if index == 0:
            new_index = other_vocab.to_indices(l[i])
            if new_index == 0: # tests for literals
                if l[i][0] == "'":
                    new_index = n_others + const_to_literal['char']
                elif l[i][0] == '"':
                    new_index = n_others + const_to_literal['string']
                elif l[i][0] == '.' and l[i][1].isdigit():
                    if l[i][-1] in 'fF':
                        new_index = n_others + const_to_literal['float']
                    else:
                        new_index = n_others + const_to_literal['double']
                elif l[i][0].isdigit():
                    if any([c in set('fF') for c in l[i]]):
                        new_index = n_others + const_to_literal['float']
                    elif any([c in set('eEdD') for c in l[i]]):
                        new_index = n_others + const_to_literal['double']
                    elif len(l[i]) > 2 and l[i][0] == '0' and l[i][1] in 'xX':
                        new_index = n_others + const_to_literal['hex_int']
                    elif len(l[i]) > 2 and l[i][0] == '0' and l[i][1] in 'bB':
                        new_index = n_others + const_to_literal['bin_int']
                    else:
                        new_index = n_others + const_to_literal['int']
                
            indices[i] = n_subtoks + new_index
        else:
            indices[i] -= 1
    return indices

def to_tokens(l): # non-reversible, i.e. to_ids(to_tokens(l)) will probably not work
    translation = [None for x in l]
    for i, x in enumerate(l):
        if x >= n_subtoks + n_others:
            translation[i] = '<<' + literal_constants[x - n_subtoks - n_others] + '_literal>>'
        elif x >= n_subtoks:
            translation[i] = other_vocab.to_tokens(x - n_subtoks)
        else:
            translation[i] = subtok_vocab.to_tokens(x + 1)
    return translation

fin = open('java_names/output.json', 'r')

save_folder = 'data/test/'
context_files = {}
random.shuffle(file_seeks)

max_uses = 100000
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
    
    fout.write(struct.pack('<2I', n_contexts, to_ids([data['variableName']])[0]))
    for context in data['usage']:
        context_a = to_subtokenized_list(context[:64])[-64:]
        context_b = to_subtokenized_list(context[129:64:-1])[-64:]
        fout.write(struct.pack('<64I', *to_ids(context_a)))
        fout.write(struct.pack('<64I', *to_ids(context_b)))
        print("-" * 100)
        #for o_token, t_token in zip(context_a[:64], to_tokens(to_ids(context_a[:64]))):
        #    print(o_token, '-->', t_token)
        #break
    bar.update(1)
    
    if seek_i == max_uses - 1:
        break

bar.close()
fin.close()

for fout in context_files.values():
    fout.close()

# idea: use first, second, third moments of the distributions of weights?
