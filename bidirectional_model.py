import mxnet as mx

class BidirectionalModel(mx.gluon.Block):
    def __init__(self, vocab, hidden_per_side, num_layers, embedding_size,
                 dropout = 0.5, tie_weights = False, ctx = mx.gpu(), **kwargs):
        super(BidirectionalModel, self).__init__(**kwargs)
        self.ctx = ctx
        self.hidden_per_side = hidden_per_side
        self.hidden_full = 2 * hidden_per_side
        self.drop = mx.gluon.nn.Dropout(dropout)
        self.embedder = mx.gluon.nn.Embedding(vocab.total_tokens, embedding_size, weight_initializer = mx.init.Uniform(0.1))
        self.forward_rnn = mx.gluon.rnn.GRU(hidden_per_side, num_layers, dropout = dropout, input_size = embedding_size, prefix = 'forward_rnn_')
        self.backward_rnn = mx.gluon.rnn.GRU(hidden_per_side, num_layers, dropout = dropout, input_size = embedding_size, prefix = 'backward_rnn_')
        self.output_rnn = mx.gluon.rnn.GRU(self.hidden_full, num_layers, dropout = dropout, input_size = embedding_size, prefix = 'output_rnn_')
        if tie_weights:
            self._contract_hiddens = mx.gluon.nn.Dense(hidden_per_side, in_units = self.hidden_full)
            self._decoder = mx.gluon.nn.Dense(vocab.total_tokens, in_units = hidden_per_side, params = self.embedder.params)
            self.decoder = lambda x: self._decoder(self._contract_hiddens(x))
        else:
            self.decoder = mx.gluon.nn.Dense(vocab.total_tokens, in_units = self.hidden_full)
    
    def forward(self, forward_contexts, backward_contexts, predict_in):
        batch_size = predict_in.shape[1]
        n_contexts = forward_contexts.shape[0]
        
        # TODO(Ben) also add in option for variance?
        f_hiddens, b_hiddens = [], []
        for ci in range(n_contexts):
            f_embed = self.drop(self.embedder(forward_contexts[ci]))
            b_embed = self.drop(self.embedder(backward_contexts[ci]))
        
            f_hidden = self.forward_rnn.begin_state(func = mx.nd.zeros, batch_size = batch_size, ctx = self.ctx)
            _, f_hidden = self.forward_rnn(f_embed, f_hidden)
            b_hidden = self.backward_rnn.begin_state(func = mx.nd.zeros, batch_size = batch_size, ctx = self.ctx)
            _, b_hidden = self.backward_rnn(f_embed, b_hidden)
            
            f_hiddens.append(f_hidden[0]) # always seems to be of length 1. Is there an edge case?
            b_hiddens.append(b_hidden[0])
        f_hidden_sum = mx.nd.add_n(*f_hiddens)
        b_hidden_sum = mx.nd.add_n(*b_hiddens)
        # combined_hidden = [mx.nd.concatenate((f_h, b_h)) / n_contexts for f_h, b_h in zip(avg_f_hidden, avg_b_hidden)]
        combined_hidden = mx.nd.concat(f_hidden_sum, b_hidden_sum, dim = 2) / n_contexts
        
        predict_embed = self.drop(self.embedder(predict_in))
        output, _ = self.output_rnn(predict_embed, combined_hidden)
        output = self.drop(output)
        output_decoded = self.decoder(output.reshape((-1, self.hidden_full)))
        return output_decoded
