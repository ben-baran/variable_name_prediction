import re
import pickle
import mxnet.contrib.text as mtext
import glob
import os
import numpy as np
import struct
import json
import mxnet as mx

with open('data_options.json') as options_f:
    options = json.load(options_f)

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

def rough_print(toks, indent_level = 3, open_indent = True):
    # start off with an extra few indent levels for safety
    if open_indent:
        print("\t" * indent_level, end = '')
    for tok_i in range(len(toks) - 1):
        if toks[tok_i + 1] == '}':
            indent_level -= 1
        if is_token(toks[tok_i]) and is_token(toks[tok_i + 1]):
            print(toks[tok_i], end = '_')
        elif toks[tok_i] in '{};':
            if toks[tok_i] == '{':
                indent_level += 1
            print(toks[tok_i])
            print("\t" * indent_level, end = '')
        elif toks[tok_i] == '<<PAD>>':
            continue
        elif toks[tok_i] == '<<end_id>>':
            if is_token(toks[tok_i + 1]):
                print(' ', end = '')
        else:
            print(toks[tok_i], end = ' ')
    if toks[-1] != '<<PAD>>' and toks[-1] != '<<end_id>>':
        print(toks[-1], end = '')
    return indent_level

class ComboVocab:
    literal_constants = ['char', 'string', 'float', 'double', 'hex_int', 'bin_int', 'int', 'start_id', 'end_id']
    const_to_literal = {s:i for i, s in enumerate(literal_constants)}
    
    def __init__(self, vocabs_fname = 'data/vocab4096.pkl', counters_fname = 'data/counters.pkl', min_for_vocabulary = 4096):
        if not os.path.isfile(vocabs_fname):
            print("Vocabulary not cached, have to build...")
            with open(counters_fname, 'rb') as counters_file:
                subtok_counter, other_counter = pickle.load(counters_file)
            subtok_counter['_'] = min_for_vocabulary
            subtok_counter['__'] = min_for_vocabulary # I had forgotten that these can exist on their own
            self.subtok_vocab = mtext.vocab.Vocabulary(subtok_counter, min_freq = min_for_vocabulary)
            self.other_vocab = mtext.vocab.Vocabulary(other_counter, min_freq = min_for_vocabulary)
            with open(vocabs_fname, 'wb') as vocabs_file:
                pickle.dump((self.subtok_vocab, self.other_vocab), vocabs_file)
        else:
            with open(vocabs_fname, 'rb') as vocabs_file:
                self.subtok_vocab, self.other_vocab = pickle.load(vocabs_file)
            
        self.n_subtoks = len(self.subtok_vocab.token_to_idx) - 1 # remove one for unknowns
        self.n_others = len(self.other_vocab.token_to_idx) # unknown remains
        self.total_tokens = self.n_subtoks + self.n_others + len(ComboVocab.literal_constants)
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
    def __init__(self, vocab, load_from = None,
                 folder_name = 'data/train_small/', batch_size = 32,
                 context_width = options['context_width'], max_subtokens_predicted = options['max_subtokens_predicted'],
                 ctx = mx.cpu()):
        self.start_token_id = vocab.to_ids(['<<start_id>>'])[0]
        self.end_token_id = vocab.to_ids(['<<end_id>>'])[0]
        self.pad_token_id = vocab.to_ids(['<<PAD>>'])[0]
        self.ctx = ctx
        
        if load_from is not None:
            self.load_state(load_from)
            return
        
        self.batch_size = batch_size
        self.context_width = context_width
        self.max_subtokens_predicted = max_subtokens_predicted
        self.context_files = []
        self.context_props = []
        for filename in os.listdir(folder_name):
            if filename[-4:] != '.bin':
                continue
            n_contexts = int(filename.split('.')[0])
            if n_contexts == 0: # for some reason this sometimes gets outputted
                continue
            full_path = folder_name + filename
            
            # we get the size for proportionally sampling the files
            size = os.path.getsize(full_path)
            if size == 0:
                continue # must be some residue from a cancelled run
            
            self.context_props.append(size / n_contexts)
            self.context_files.append((n_contexts, open(full_path, 'rb')))
        self.context_props = np.array(self.context_props) / np.sum(self.context_props)
        
    def save_state(self, filename):
        surrogate = {}
        surrogate['batch_size'] = self.batch_size
        surrogate['context_width'] = self.context_width
        surrogate['max_subtokens_predicted'] = self.max_subtokens_predicted
        surrogate['context_files'] = []
        for n_contexts, file in self.context_files:
            surrogate['context_files'].append((n_contexts, file.name, file.tell()))
        surrogate['context_props'] = self.context_props
        with open(filename, 'wb') as pickle_file:
            pickle.dump(surrogate, pickle_file)
        
    
    def load_state(self, load_from):
        with open(load_from, 'rb') as pickle_file:
            surrogate = pickle.load(pickle_file)
        self.batch_size = surrogate['batch_size']
        self.context_width = surrogate['context_width']
        self.max_subtokens_predicted = surrogate['max_subtokens_predicted']
        self.context_files = []
        for n_contexts, filename, position in surrogate['context_files']:
            f = open(filename, 'rb')
            f.seek(position)
            self.context_files.append((n_contexts, f))
        self.context_props = surrogate['context_props']
            
    def iterator(self):
        # for possible future work in making this run on a separate thread with a Queue
        batch = self.get_batch()
        while batch is not None:
            yield batch
            batch = self.get_batch()
    
    def get_batch(self):
        """
        Returns random n_contexts of [pre_context], [post_context], and batched input_vars, target_vars, target_mask
        
        Input vars is something like [<BEGIN> a b c]
        Output vars is something like [a b c <END>]
        If one in the batch is longer than others, pad the rest.
        
        Inputs have the shape [time, batch]
        Outputs are of the form [time, batch].reshape((-1,))
        """
        while len(self.context_files) > 0:
            choice = np.random.choice(len(self.context_files), p = self.context_props)
            n_contexts, fin = self.context_files[choice]
            try:
                return self._try_load(n_contexts, fin)
            except struct.error: # we've run out of room in the file. Hangers-on will be left behind.
                self.context_files[choice][1].close()
                del self.context_files[choice]
                self.context_props = np.delete(self.context_props, choice)
                if self.context_props.size > 0:
                    self.context_props /= np.sum(self.context_props)
        return None
    
    def _try_load(self, n_contexts, fin):
        # A helper function for get_batch. Will raise an error when file runs out of space.
        pre_contexts = [np.zeros((self.batch_size, self.context_width)) for i in range(n_contexts)]
        post_contexts = [np.zeros((self.batch_size, self.context_width)) for i in range(n_contexts)]
        input_tokens = []
        output_tokens = []
        for batch_i in range(self.batch_size):
            usage = list(struct.unpack('<%dI' % self.max_subtokens_predicted, fin.read(4 * self.max_subtokens_predicted)))
            try:
                usage = usage[:next((i for i in range(len(usage)) if usage[i] == self.pad_token_id), len(usage))]
            except StopIteration:
                pass
            input_tokens.append([self.start_token_id] + usage)
            output_tokens.append(usage + [self.end_token_id])
            for context_n in range(n_contexts):
                read_pre = struct.unpack('<%dI' % self.context_width, fin.read(4 * self.context_width))
                read_post = struct.unpack('<%dI' % self.context_width, fin.read(4 * self.context_width))
                pre_contexts[context_n][batch_i] = read_pre
                post_contexts[context_n][batch_i] = read_post
        longest_var = max([len(toks) for toks in input_tokens])
        input_tokens = [u + [self.pad_token_id for i in range(longest_var - len(u))] for u in input_tokens]
        output_tokens = [u + [self.pad_token_id for i in range(longest_var - len(u))] for u in output_tokens]
        input_tokens = mx.nd.array(input_tokens, ctx = self.ctx).T
        
        output_tokens = np.array(output_tokens).T.reshape((-1,))
        target_mask = mx.nd.array([float(tok != self.pad_token_id) for tok in output_tokens], ctx = self.ctx)
        output_tokens = mx.nd.array(output_tokens, ctx = self.ctx)
        
        pre_contexts = mx.nd.array(pre_contexts, ctx = self.ctx).transpose((1, 2, 0))
        post_contexts = mx.nd.array(post_contexts, ctx = self.ctx).transpose((1, 2, 0))

        return pre_contexts.T, post_contexts.T, input_tokens, output_tokens, target_mask
