import re
import pickle
import mxnet.contrib.text as mtext
import glob
import os
import numpy as np
import struct

kw_or_builtin = set(('abstract continue for new switch assert default goto package synchronized '
     'boolean do if private this break double implements protected throw byte else '
     'import public throws case enum instanceof return transient catch extends int '
     'short try char final interface static void class finally long strictfp '
     'volatile const float native super while true false null').split(' '))

def is_token(s):
    if not (s[0].isalpha() or s[0] in '$_'):
        return False
    return s not in kw_or_builtin

def subtokenize(token):
    c_style = [st for st in token.split('_') if len(st) > 0]
    subtokens = []
    for subtoken in c_style:
        # from https://stackoverflow.com/questions/29916065/how-to-do-camelcase-split-in-python
        camel_splits = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', subtoken)
        subtokens += [split.group(0).lower() for split in camel_splits]
    return subtokens

def to_subtokenized_list(l):
    stized = []
    for item in l:
        if is_token(item):
            subtokens = subtokenize(item)
            if len(subtokens) == 0:
                stized.append(item)
            else:
                stized.extend(subtokenize(item))
            stized.append('<<end_id>>')
        else:
            stized.append(item)
    return stized

class ComboVocab:
    literal_constants = ['char', 'string', 'float', 'double', 'hex_int', 'bin_int', 'int', 'start_id', 'end_id']
    const_to_literal = {s:i for i, s in enumerate(literal_constants)}
    
    def __init__(self, counters_fname = 'data/counters.pkl', min_for_vocabulary = 4096):
        with open(counters_fname, 'rb') as counter_file:
            self.subtok_counter, self.other_counter = pickle.load(counter_file)
        self.subtok_counter['_'] = min_for_vocabulary
        self.subtok_counter['__'] = min_for_vocabulary # I had forgotten that these can exist on their own
        self.subtok_vocab = mtext.vocab.Vocabulary(self.subtok_counter, min_freq = min_for_vocabulary)
        self.n_subtoks = len(self.subtok_vocab.token_to_idx) - 1 # remove one for unknowns
        self.other_vocab = mtext.vocab.Vocabulary(self.other_counter, min_freq = min_for_vocabulary)
        self.n_others = len(self.other_vocab.token_to_idx) # unknown remains
        print("Number of subtokens: %d, number of other tokens: %d" % (self.n_subtoks, self.n_others - 1))
    
    def to_ids(self, l):
        indices = self.subtok_vocab.to_indices(l)
        for i, index in enumerate(indices):
            if index == 0:
                new_index = self.other_vocab.to_indices(l[i])
                if new_index == 0: # tests for literals
                    removed_brackets = l[i][2:-2]
                    if removed_brackets in ComboVocab.const_to_literal:
                        new_index = self.n_others + ComboVocab.const_to_literal[removed_brackets]
                    elif l[i][0] == "'":
                        new_index = self.n_others + ComboVocab.const_to_literal['char']
                    elif l[i][0] == '"':
                        new_index = self.n_others + ComboVocab.const_to_literal['string']
                    elif l[i][0] == '.' and l[i][1].isdigit():
                        if l[i][-1] in 'fF':
                            new_index = self.n_others + ComboVocab.const_to_literal['float']
                        else:
                            new_index = self.n_others + ComboVocab.const_to_literal['double']
                    elif l[i][0].isdigit():
                        if any([c in set('fF') for c in l[i]]):
                            new_index = self.n_others + ComboVocab.const_to_literal['float']
                        elif any([c in set('eEdD') for c in l[i]]):
                            new_index = self.n_others + ComboVocab.const_to_literal['double']
                        elif len(l[i]) > 2 and l[i][0] == '0' and l[i][1] in 'xX':
                            new_index = self.n_others + ComboVocab.const_to_literal['hex_int']
                        elif len(l[i]) > 2 and l[i][0] == '0' and l[i][1] in 'bB':
                            new_index = self.n_others + ComboVocab.const_to_literal['bin_int']
                        else:
                            new_index = self.n_others + ComboVocab.const_to_literal['int']

                indices[i] = self.n_subtoks + new_index
            else:
                indices[i] -= 1
        return indices
    
    def to_tokens(self, l):
        translation = [None for x in l]
        for i, x in enumerate(l):
            if x >= self.n_subtoks + self.n_others:
                translation[i] = '<<' + ComboVocab.literal_constants[x - self.n_subtoks - self.n_others] + '>>'
            elif x >= self.n_subtoks:
                translation[i] = self.other_vocab.to_tokens(x - self.n_subtoks)
            else:
                translation[i] = self.subtok_vocab.to_tokens(x + 1)
        return translation

class ContextLoader:
    def __init__(self, folder_name = 'data/train_tmp/', batch_size = 32):
        self.batch_size = batch_size
        self.context_files = []
        self.context_props = []
        for filename in os.listdir(folder_name):
            if filename[-4:] != '.bin':
                continue
            n_contexts = int(filename.split('.')[0])
            full_path = folder_name + filename
            
            # we get the size for proportionally sampling the files
            size = os.path.getsize(full_path)
            self.context_props.append(size)
            self.context_files.append((n_contexts, open(full_path, 'rb')))
        total_size = sum(self.context_props)
        self.context_props = np.array([size / total_size for size in self.context_props])
            
    
    def get_batch(self):
        # returns random n_contexts of [pre_sequence, post_sequence, input_vars, output_vars]
        # input vars is something like [<BEGIN> a b c]
        # output vars is something like [a b c <END>]
        # if one in the batch is longer than others, pad the rest
        choice = np.random.choice(len(self.context_files), p = self.context_props)
        n_contexts, fin = self.context_files[choice]
        predict_subtokens = struct.unpack('<8I', fin.read(32))
        pre_contexts, post_contexts = [], []
        for context_n in range(n_contexts):
            pre_contexts.append(struct.unpack('<64I', fin.read(256)))
            post_contexts.append(struct.unpack('<64I', fin.read(256)))
        return predict_subtokens, pre_contexts, post_contexts