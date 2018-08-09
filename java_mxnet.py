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

argument_parser = argparse.ArgumentParser(description = "Train a token predictor for Java")

argument_parser.add_argument("-sdp", "--save-dir-prefix", default = './search_simple/run', type = str,
                help = "Directory prefix to save data to. Must not already exist.")

argument_parser.add_argument("-e", "--epochs", default = 1, type = int,
                help = "Number of epochs to train")

argument_parser.add_argument("-b", "--batch-size", default = 32, type = int,
                help = "Number of frame groups to process at a time")

argument_parser.add_argument("-nh", "--num-hidden", default = 256, type = int,
                help = "Number of hidden weights per GRU layer")

argument_parser.add_argument("-nl", "--num-layers", default = 2, type = int,
                help = "Number of GRU layers")

argument_parser.add_argument("--token-embed-size", default = 240, type = int,
                help = "Size of token embeddings")

argument_parser.add_argument("--type-embed-size", default = 16, type = int,
                help = "Size of type embeddings")

argument_parser.add_argument("-t", "--time", default = 256, type = int,
                help = "How many tokens to backpropagate error to in the past")

argument_parser.add_argument("--clip", default = 0.2, type = float,
                help = "Amount to clip weights by.")

argument_parser.add_argument("-lr", "--learning-rate", default = 1e-3, type = float,
                help = "Learning rate for Adam.")

argument_parser.add_argument("--lr-decay", default = 1.0, type = float,
                help = "Factor to multiply learning rate by every epoch.")

argument_parser.add_argument("--time-limit", default = 1000000, type = int,
                help = "Maximum number of seconds this can train.")


options = argument_parser.parse_args()

save_dir = options.save_dir_prefix + datetime.datetime.now().strftime('.%Y.%m.%d.%H.%M.%S/')
if os.path.exists(save_dir):
    print('Save directory already exists!')
    quit()

os.makedirs(save_dir)
shutil.copyfile(__file__, save_dir + 'java_mxnet.py')
with open(save_dir + 'options.json', 'w') as options_file:
    json.dump(vars(options), options_file, indent = 1)

ctx = mx.gpu(3)
batch_size = options.batch_size
time_slice = options.time
clip = options.clip

with open('java_token_vocab.pickle', 'rb') as f:
    token_vocab = pickle.load(f)
with open('java_type_vocab.pickle', 'rb') as f:
    type_vocab = pickle.load(f)
n_tokens = len(token_vocab.token_to_idx)
n_types = len(type_vocab.token_to_idx)

print("Number of tokens: %d, number of types: %d" % (n_tokens, n_types))

class VocabModel(mx.gluon.Block):
    def __init__(self, num_hidden, num_layers, token_embed_size, type_embed_size,
                 dropout = 0.5, tie_weights = False, **kwargs):
        super(VocabModel, self).__init__(**kwargs)
        self.num_hidden = num_hidden
        self.drop = mx.gluon.nn.Dropout(dropout)
        self.tok_encoder = mx.gluon.nn.Embedding(n_tokens, token_embed_size, weight_initializer = mx.init.Uniform(0.1))
        self.type_encoder = mx.gluon.nn.Embedding(n_types, type_embed_size, weight_initializer = mx.init.Uniform(0.1))
        self.rnn = mx.gluon.rnn.GRU(num_hidden, num_layers, dropout = dropout, input_size = token_embed_size + type_embed_size)
        if tie_weights:
            self.decoder = mx.gluon.nn.Dense(n_tokens + n_types, in_units = num_hidden, params = self.encoder.params)
        else:
            self.decoder = mx.gluon.nn.Dense(n_tokens + n_types, in_units = num_hidden)
    
    def forward(self, token_inputs, type_inputs, hidden):
        token_embed = self.drop(self.tok_encoder(token_inputs))
        type_embed = self.drop(self.type_encoder(type_inputs))
        total_embed = mx.nd.concat(token_embed, type_embed, dim = 2)
        output, hidden = self.rnn(total_embed, hidden)
        output = self.drop(output)
        total_decoded = self.decoder(output.reshape((-1, self.num_hidden))) 
        tokens_decoded = total_decoded.slice_axis(axis = 1, begin = 0, end = n_tokens)
        types_decoded = total_decoded.slice_axis(axis = 1, begin = n_tokens, end = None)
        return tokens_decoded, types_decoded, hidden
    
    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


# x:x+time -> x+1:x+1+time
def batches_gen(f_prefix, time_slice = 256, batch_size = 32, ctx = mx.gpu(0)):
    data = open(f_prefix + '.bin', 'rb')
    with open(f_prefix + "_starts.pickle", 'rb') as f:
        starts = pickle.load(f)
    batch_starts = []
    for i in range(len(starts) - 1):
        batch_starts.extend(list(range(starts[i], starts[i + 1] - time_slice - 1)))
    np.random.shuffle(batch_starts)
    
    batch_i = 0
    while batch_i < len(batch_starts):
        tokens_in, tokens_target, types_in, types_target = [np.zeros((batch_size, time_slice)) for i in range(4)]
        for i in range(batch_size):
            data_i = batch_starts[batch_i]
            data.seek(data_i * 2)
            slice_A = struct.unpack('ll', data.read(2)))
            slice_B = struct.unpack('ll', data.read(2)))
            for slice_i in range(time_slice):
                tokens_in[i, slice_i] = slice_A[0]
                types_in[i, slice_i] = slice_A[1]
                tokens_target[i, slice_i] = slice_B[0]
                types_target[i, slice_i] = slice_B[1]
                if slice_i != time_slice - 1:
                    slice_A = slice_B
                    slice_B = struct.unpack('ll', data.read(2))
            batch_i += 1
        tokens_in = mx.nd.array(tokens_in, ctx = ctx).T
        types_in = mx.nd.array(types_in, ctx = ctx).T
        tokens_target = mx.nd.array(tokens_target, ctx = ctx).T.reshape((-1,))
        types_target = mx.nd.array(types_target, ctx = ctx).T.reshape((-1,))
        yield tokens_in, types_in, tokens_target, types_target
    data.close()
    
def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden

model = VocabModel(num_hidden = options.num_hidden,
                   num_layers = options.num_layers,
                   token_embed_size = options.token_embed_size,
                   type_embed_size = options.type_embed_size)
model.hybridize()
model.collect_params().initialize(mx.init.Xavier(), ctx = ctx)
trainer = mx.gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': options.learning_rate})
loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()

def create_prediction(num = 16, tokens = ['public', 'class', 'HTTP', '{', 'public', 'static', 'void'],
                      types = ["<'public'>", "<'class'>", "<Identifier>", "<'{'>}", "<'public'>", "<'static'>", "<'void'>"]):
    tokens_in, types_in = mx.nd.zeros((1, batch_size), ctx = ctx), mx.nd.zeros((1, batch_size), ctx = ctx)
    hidden = model.begin_state(func = mx.nd.zeros, batch_size = batch_size, ctx = ctx)
    for i, (token, type_str) in enumerate(zip(tokens, types)):
        tokens_in[0, 0] = token_vocab.to_indices(token)
        types_in[0, 0] = type_vocab.to_indices(type_str)
        tokens_out, types_out, hidden = model(tokens_in, types_in, hidden)
    print(' '.join(tokens), end = ' ')
    
    for i in range(num):
        best_token = np.argmax(tokens_out.reshape((1, batch_size, n_tokens))[0, 0, :].asnumpy())
        best_type = np.argmax(types_out.reshape((1, batch_size, n_types))[0, 0, :].asnumpy())
        print(token_vocab.to_tokens(int(best_token)), end = ' ')
        tokens_out, types_out, hidden = model(tokens_in, types_in, hidden)
        tokens_in[0, 0] = best_token
        types_in[0, 0] = best_type
    print()

def most_likely(num = 16, tokens = ['public', 'class', 'HTTP', '{', 'public', 'static', 'void'],
                types = ["<'public'>", "<'class'>", "<Identifier>", "<'{'>}", "<'public'>", "<'static'>", "<'void'>"]):
    tokens_in, types_in = mx.nd.zeros((1, batch_size), ctx = ctx), mx.nd.zeros((1, batch_size), ctx = ctx)
    hidden = model.begin_state(func = mx.nd.zeros, batch_size = batch_size, ctx = ctx)
    for i, (token, type_str) in enumerate(zip(tokens, types)):
        tokens_in[0, 0] = token_vocab.to_indices(token)
        types_in[0, 0] = type_vocab.to_indices(type_str)
        tokens_out, types_out, hidden = model(tokens_in, types_in, hidden)
    tokens_out = tokens_out.reshape((1, batch_size, n_tokens))[0, 0, :].asnumpy()
    top = np.argsort(tokens_out)[-num:][::-1]
    print("Most likely:")
    for i in range(num):
        print(token_vocab.to_tokens(int(top[i])), '(%f)' % tokens_out[top[i]])
        
test_batches = batches_gen('mx_datasets/test', time_slice = time_slice, batch_size = batch_size, ctx = ctx)

def validation_loss(num_minibatches = 1):
    global test_batches
    i = 0
    avg_loss = 0.0
    for tokens_in, types_in, tokens_target, types_target in test_batches:
        i += 1
        hidden = model.begin_state(func = mx.nd.zeros, batch_size = batch_size, ctx = ctx)
        tokens_output, types_output, hidden = model(tokens_in, types_in, hidden)
        L = loss(tokens_output, tokens_target) + loss(types_output, types_target)
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
    train_batches = batches_gen('mx_datasets/train', time_slice = time_slice, batch_size = batch_size, ctx = ctx)
    for tokens_in, types_in, tokens_target, types_target in train_batches:
        hidden = model.begin_state(func = mx.nd.zeros, batch_size = batch_size, ctx = ctx)
        # hidden = detach(hidden)
        with mx.autograd.record():
            tokens_output, types_output, hidden = model(tokens_in, types_in, hidden)
            L = loss(tokens_output, tokens_target) + loss(types_output, types_target)
            L.backward()
        grads = [i.grad(ctx) for i in model.collect_params().values()]
        mx.gluon.utils.clip_global_norm(grads, clip * time_slice * batch_size)
        trainer.step(batch_size)
        validation_losses.append(validation_loss())
        if i % 500 == 0:
            print("i = %d. Saving." % i)
            model.save_params(save_dir + 'epoch-%d-i-%.6d.params' % (epoch, i))
            np.save(save_dir + 'validation_losses', np.array(validation_losses))
            create_prediction()
            most_likely()
        if time.time() - start_global > options.time_limit:
            print("Time limit reached. Ending epoch.")
            break
        i += 1
    print("Epoch completed. %d iterations" % i)
    if time.time() - start_global > options.time_limit:
        break
