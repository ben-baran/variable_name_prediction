import random
import pickle
import struct
import json
import mxnet.contrib.text as mtext
from tqdm import tqdm

min_for_vocabulary = 4096
n_ctx = 64 # size of context per side

with open('data/counters_000200000.pkl', 'rb') as counter_file:
    subtok_counter, other_counter = pickle.load(counter_file)
with open('data/json_seeks_000200000.pkl', 'rb') as seeks_file:
    file_seeks = pickle.load(seeks_file)

subtok_vocab = mtext.vocab.Vocabulary(subtok_counter, min_freq = min_for_vocabulary)
n_subtoks = len(subtok_vocab.token_to_idx) - 1 # remove one for unknowns
other_vocab = mtext.vocab.Vocabulary(other_counter, min_freq = min_for_vocabulary)

print("Number of subtokens: %d, number of other tokens: %d" % (n_subtoks, len(other_vocab.token_to_idx) - 1))

def to_ids(l):
    indices = subtok_vocab.to_indices(l)
    for i, index in enumerate(indices):
        if index == 0:
            indices[i] = n_subtoks + other_vocab.to_indices(index)
        else:
            indices[i] -= 1
    return indices

fin = open('java_names/output.json', 'r')
fout = open('data/contexts_min512.bin', 'wb')
random.shuffle(file_seeks)

bar = tqdm(total = len(file_seeks), desc = 'processing', unit = 'ctx groups')
for seek in file_seeks:
    # For every set of contexts this outputs: N_CTX ID BEFORE_CTX AFTER_CTX BEFORE_CTX AFTER_CTX
    fin.seek(seek)
    line = fin.readline()
    if len(line) == 0:
        continue
    data = json.loads(line)
    n_contexts = len(data['usage'])
    fout.write(struct.pack('<2I', n_contexts, to_ids([data['variableName']])[0]))
    for context in data['usage']:
        fout.write(struct.pack('<64I', *to_ids(context[:64])))
        fout.write(struct.pack('<64I', *to_ids(context[129:64:-1])))
    bar.update(1)

bar.close()
fin.close()
fout.close()
# idea: use first, second, third moments of the distributions of weights?
