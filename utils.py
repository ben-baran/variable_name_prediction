import re
import pickle
import mxnet.contrib.text as mtext

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
        else:
            stized.append(item)
    return stized

class ComboVocab:
    literal_constants = ['char', 'string', 'float', 'double', 'hex_int', 'bin_int', 'int']
    const_to_literal = {s:i for i, s in enumerate(literal_constants)}
    
    def __init__(self, counters_fname = 'data/counters.pkl', min_for_vocabulary = 4096):
        with open(counters_fname, 'rb') as counter_file:
            self.subtok_counter, self.other_counter = pickle.load(counter_file)
        self.subtok_counter['_'] = 4096
        self.subtok_counter['__'] = 4096 # I had forgotten that these can exist on their own
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
                    if l[i][0] == "'":
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
    
    def to_tokens(self, l): # non-reversible, i.e. to_ids(to_tokens(l)) will probably not work
        translation = [None for x in l]
        for i, x in enumerate(l):
            if x >= self.n_subtoks + self.n_others:
                translation[i] = '<<' + ComboVocab.literal_constants[x - self.n_subtoks - self.n_others] + '_literal>>'
            elif x >= self.n_subtoks:
                translation[i] = self.other_vocab.to_tokens(x - self.n_subtoks)
            else:
                translation[i] = self.subtok_vocab.to_tokens(x + 1)
        return translation
