from bidirectional_model import BidirectionalModel
from utils import ComboVocab
import mxnet as mx
import json
from easydict import EasyDict

class VariablePredictor:
    def __init__(self, params_filename = 'with_prefix/run.2018.10.11.12.03.10/epoch-0-000011.params',
                       run_options = 'with_prefix/run.2018.10.11.12.03.10/run_options.json',
                       vocabs_fname = 'data/vocab4096.pkl',
                       counters_fname = 'data/counters.pkl',
                       gpu_to_use = 2):
        with open(run_options, 'r') as options_file:
            options = EasyDict(json.load(options_file))
        self.vocab = ComboVocab(vocabs_fname = vocabs_fname, counters_fname = counters_fname)
        self.ctx = mx.gpu(gpu_to_use)
        self.model = BidirectionalModel(self.vocab,
                                        hidden_per_side = options.hidden_per_side,
                                        num_layers = options.num_layers,
                                        embedding_size = options.embedding_size,
                                        dropout = options.dropout,
                                        tie_weights = options.tie_weights,
                                        ctx = self.ctx)
        self.model.hybridize()
        self.model.collect_params().load(params_filename, self.ctx)
        
    def predict(self, tokens_before, tokens_after, top_k = 5):
        # transform tokens into their IDs
        return self.model.beam_search(tokens_before, tokens_after, self.vocab, top_k)
    
# EXAMPLE
# still need to implement the search and transforming the IDs
# need to find example text
# parameters aren't loading correctly
var_predict = VariablePredictor()
with open('java_names/output.json', 'r') as fin:
    s = fin.readline()
print(s)
# print(var_predict.predict())