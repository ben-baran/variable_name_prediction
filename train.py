import mxnet as mx
import numpy as np
import os
import pickle
import argparse
import datetime
import shutil
import json
import struct
import time
from utils import ContextLoader, ComboVocab

argument_parser = argparse.ArgumentParser(description = "Train a variable name predictor for Java")

argument_parser.add_argument("-sdp", "--save-dir-prefix", default = './search/run', type = str,
                help = "Directory prefix to save data to. Must not already exist.")

argument_parser.add_argument("-e", "--epochs", default = 1, type = int,
                help = "Number of epochs to train")

argument_parser.add_argument("-b", "--batch-size", default = 32, type = int,
                help = "Number of frame groups to process at a time")

argument_parser.add_argument("-nh", "--num-hidden-per-side", default = 256, type = int,
                help = "Number of hidden weights per GRU layer in the forward/backward networks")

argument_parser.add_argument("-nl", "--num-layers", default = 2, type = int,
                help = "Number of GRU layers")

argument_parser.add_argument("--embedding-size", default = 256, type = int,
                help = "Size of embeddings")

argument_parser.add_argument("--clip", default = 0.2, type = float,
                help = "Amount to clip weights by.")

argument_parser.add_argument("-lr", "--learning-rate", default = 1e-3, type = float,
                help = "Learning rate for Adam.")

argument_parser.add_argument("--lr-decay", default = 1.0, type = float,
                help = "Factor to multiply learning rate by every epoch.")

argument_parser.add_argument("--time-limit", default = int(1e12), type = int,
                help = "Maximum number of seconds this can train.")


options = argument_parser.parse_args()

with open('data_options.json') as data_options_f:
    data_options = json.load(data_options_f)

save_dir = options.save_dir_prefix + datetime.datetime.now().strftime('.%Y.%m.%d.%H.%M.%S/')
if os.path.exists(save_dir):
    print('Save directory already exists!')
    quit()

os.makedirs(save_dir)
shutil.copyfile(__file__, save_dir + 'train.py')
with open(save_dir + 'run_options.json', 'w') as options_file:
    json.dump(vars(options), options_file, indent = 1)

ctx = mx.gpu(3)
vocab = ComboVocab()

class BidirectionalModel(mx.gluon.Block):
    def __init__(self, vocab, hidden_per_side, num_layers, embedding_size,
                 dropout = 0.5, tie_weights = False, **kwargs):
        super(BidirectionalModel, self).__init__(**kwargs)
        self.hidden_per_side = hidden_per_side
        self.hidden_full = 2 * hidden_per_side
        self.drop = mx.gluon.nn.Dropout(dropout)
        self.embedder = mx.gluon.nn.Embedding(vocab.total_tokens, embedding_size, weight_initializer = mx.init.Uniform(0.1))
        self.forward_rnn = mx.gluon.rnn.GRU(hidden_per_side, num_layers, dropout = dropout, input_size = embedding_size)
        self.backward_rnn = mx.gluon.rnn.GRU(hidden_per_side, num_layers, dropout = dropout, input_size = embedding_size)
        self.output_rnn = mx.gluon.rnn.GRU(self.hidden_full, num_layers, dropout = dropout, input_size = embedding_size)
        if tie_weights:
            contract_hiddens = mx.gluon.nn.Dense(hidden_per_side, in_units = self.hidden_full)
            decoder = mx.gluon.nn.Dense(vocab.total_tokens, in_units = hidden_per_side, params = self.encoder.params)
            self.decoder = lambda x: decoder(contract_hiddens(x))
        else:
            decoder = mx.gluon.nn.Dense(vocab.total_tokens, in_units = self.hidden_full)
    
    def forward(self, forward_contexts, backward_contexts, predict_in, is_training = True):
        dropout_mode = 'training' if is_training else 'always'
        batch_size = predict_in.shape[1]
        n_contexts = forward_contexts.shape[0]
        
        # TODO(Ben) also add in option for variance?
        avg_f_hidden, avg_b_hidden = None, None
        for ci in range(n_contexts):
            f_embed = self.drop(self.embedder(forward_contexts[ci]), mode = dropout_mode)
            b_embed = self.drop(self.embedder(backward_contexts[ci]), mode = dropout_mode)
        
            f_hidden = self.forward_rnn.begin_state(func = mx.nd.zeros, batch_size = batch_size, ctx = ctx)
            _, f_hidden = self.forward_rnn(f_embed, f_hidden)
            b_hidden = self.nackward_rnn.begin_state(func = mx.nd.zeros, batch_size = batch_size, ctx = ctx)
            _, b_hidden = self.forward_rnn(f_embed, b_hidden)
            
            if avg_f_hidden is None:
                avg_f_hidden = f_hidden
                avg_b_hidden = b_hidden
            else:
                for hid_i in range(len(f_hidden)):
                    avg_f_hidden[hid_i] += f_hidden[hid_i]
                    avg_b_hidden[hid_i] += b_hidden[hid_i]
        combined_hidden = [mx.nd.concatenate((f_h, b_h)) / n_contexts for f_h, b_h in zip(avg_f_hidden, avg_b_hidden)]
        
        predict_embed = self.drop(self.embedder(predict_in), mode = dropout_mode)
        output, _ = self.output_rnn(predict_embed, combined_hidden)
        output = self.drop(output, mode = dropout_mode)
        output_decoded = self.decoder(output.reshape((-1, 2 * self.num_hidden)))
        return output_decoded

model = BidirectionalModel(vocab,
                           hidden_per_side = options.hidden_per_side,
                           num_layers = options.num_layers,
                           embedding_sizee = options.embedding_size,
                           dropout = options.dropout,
                           tie_weights = options.tie_weights)
model.hybridize()
model.collect_params().initialize(mx.init.Xavier(), ctx = ctx)
trainer = mx.gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': options.learning_rate})
loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()

validation_data = ContextLoader(vocab, batch_size = options.batch_size, folder_name = 'data/val_small')
def validation_loss(num_minibatches = 10, verbose = False):
    global test_batches
    i = 0
    avg_loss = 0.0
    for pre_contexts, post_contexts, in_tokens, target_tokens in validation_data:
        i += 1
        output_tokens = model(pre_contexts, post_contexts, in_tokens)
        
        if verbose: # will only print out the first batches
            for context_i in range(pre_contexts.shape[0]):
                ids_before = [int(x) for x in pre_contexts[0, :, context_i].asnumpy()]
                ids_after = [int(x) for x in post_contexts[0, :, context_i].asnumpy()]
                print('context before:', vocab.to_tokens(ids_before))
                print('context after:', vocab.to_tokens(ids_after))
            
            out_argmax = output_tokens.reshape((-1, options.batch_size, vocab.total_tokens))[:, 0, :].argmax(axis=1).asnumpy()
            print("The most likely token stream is:", vocab.to_tokens([int(x) for x in out_argmax]))
            
        
        L = loss(output_tokens, target_tokens) # TODO(Ben): zero out the irrelevant padding tokens
        avg_loss += mx.nd.mean(L).asscalar()
        if i == num_minibatches:
            break
    if i < num_minibatches:
        test_batches = batches_gen('mx_datasets/test', time_slice = time_slice, batch_size = batch_size, ctx = ctx)
        return validation_loss(num_minibatches)
    return avg_loss / num_minibatches


validation_losses = []
start_global = time.time()
for epoch in range(options.epochs):
    i = 0
    cur_lr = options.learning_rate * options.lr_decay ** epoch
    print("\nCurrent learning rate: %f" % cur_lr)
    trainer.set_learning_rate(cur_lr)
    train_data = ContextLoader(vocab, batch_size = 32, folder_name = 'data/train_small', ctx = ctx)
    for pre_contexts, post_contexts, in_tokens, target_tokens in train_data:
        with mx.autograd.record():
            output_tokens = model(pre_contexts, post_contexts, in_tokens)
            L = loss(output_tokens, target_tokens) # TODO(Ben): zero out the irrelevant padding tokens
            L.backward()
        grads = [i.grad(ctx) for i in model.collect_params().values()]
        mx.gluon.utils.clip_global_norm(grads, options.clip * data_options['context_width'] * options.batch_size) # TODO(Ben): adjust clipping for type of network
        trainer.step(batch_size)
        
        if i % 100 == 0:
            validation_losses.append(validation_loss())
        if i % 500 == 0:
            print("i = %d. Saving." % i)
            model.save_params(save_dir + 'epoch-%d-i-%.6d.params' % (epoch, len(validation_losses)))
            validation_loss(num_minibatches = 1, verbose = True)
            np.save(save_dir + 'validation_losses', np.array(validation_losses))
        if time.time() - start_global > options.time_limit:
            print("Time limit reached. Ending epoch.")
            break
        i += 1
    print("Epoch completed. %d iterations" % i)
    if time.time() - start_global > options.time_limit:
        break
