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
from utils import ContextLoader, ComboVocab, rough_print
from bidirectional_model import BidirectionalModel

argument_parser = argparse.ArgumentParser(description = "Train a variable name predictor for Java")

argument_parser.add_argument("-sdp", "--save-dir-prefix", default = './search/run', type = str,
                help = "Directory prefix to save data to. Must not already exist.")

argument_parser.add_argument("-e", "--epochs", default = 1, type = int,
                help = "Number of epochs to train")

argument_parser.add_argument("-b", "--batch-size", default = 32, type = int,
                help = "Number of frame groups to process at a time")

argument_parser.add_argument("-nh", "--hidden-per-side", default = 256, type = int,
                help = "Number of hidden weights per GRU layer in the forward/backward networks")

argument_parser.add_argument("-nl", "--num-layers", default = 2, type = int,
                help = "Number of GRU layers")

argument_parser.add_argument("--embedding-size", default = 256, type = int,
                help = "Size of embeddings")

argument_parser.add_argument("--dropout", default = 0.5, type = float,
                help = "Amount to apply dropout.")

argument_parser.add_argument("--clip", default = 0.2, type = float,
                help = "Amount to clip weights by.")

argument_parser.add_argument("-lr", "--learning-rate", default = 1e-3, type = float,
                help = "Learning rate for Adam.")

argument_parser.add_argument("--lr-decay", default = 1.0, type = float,
                help = "Factor to multiply learning rate by every epoch.")

argument_parser.add_argument("--time-limit", default = int(1e15), type = int,
                help = "Maximum number of seconds this can train.")

argument_parser.add_argument('--tie-weights', action = 'store_true',
                    help = 'If you tie the weights of the decoder and encoder, you have to have an extra funneling step.')

argument_parser.set_defaults(tie_weights = False)
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
model = BidirectionalModel(vocab,
                           hidden_per_side = options.hidden_per_side,
                           num_layers = options.num_layers,
                           embedding_size = options.embedding_size,
                           dropout = options.dropout,
                           tie_weights = options.tie_weights,
                           ctx = ctx)
model.hybridize()
model.collect_params().initialize(mx.init.Xavier(), ctx = ctx)
trainer = mx.gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': options.learning_rate})
loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()

validation_data = ContextLoader(vocab, batch_size = options.batch_size, folder_name = 'data/val_full/', ctx = ctx)
def validation_loss(num_minibatches = 10, verbose = False):
    global validation_data
    i = 0
    avg_loss = 0.0
    for pre_contexts, post_contexts, in_tokens, target_tokens, target_mask in validation_data.iterator():
        i += 1
        output_tokens = model(pre_contexts, post_contexts, in_tokens)
        
        if verbose: # will only print out the first batches
            out_argmax = output_tokens.reshape((-1, options.batch_size, vocab.total_tokens))[:, 0, :].argmax(axis=1).asnumpy()
            
            for context_i in range(pre_contexts.shape[0]):
                if context_i > 3:
                    break # don't need too much information
                print("BEGINNING CONTEXT")
                ids_before = [int(x) for x in pre_contexts[0, :, context_i].asnumpy()]
                ids_after = [int(x) for x in post_contexts[0, :, context_i].asnumpy()]
                indent_level = rough_print(vocab.to_tokens(ids_before))
                print('\x1b[%sm' % '6;30;46', end = '') # from https://stackoverflow.com/a/21786287/3487347
                print('  ', end = '')
                indent_level = rough_print(vocab.to_tokens([int(x) for x in out_argmax]), indent_level = indent_level, open_indent = False)
                print(' ', end = '')
                print('\x1b[0m', end = '')
                rough_print(vocab.to_tokens(ids_after[::-1]), indent_level = indent_level, open_indent = False)
                print()
            
        
        L = loss(output_tokens, target_tokens) # TODO(Ben): zero out the irrelevant padding tokens
        L_masked = L * target_mask
        avg_loss += mx.nd.mean(L_masked).asscalar()
        if i == num_minibatches:
            break
    if i < num_minibatches:
        validation_data = ContextLoader(vocab, batch_size = options.batch_size, folder_name = 'data/val_full/', ctx = ctx)
        return validation_loss(num_minibatches)
    return avg_loss / num_minibatches


validation_losses = []
start_global = time.time()
for epoch in range(options.epochs):
    iteration = 0
    cur_lr = options.learning_rate * options.lr_decay ** epoch
    print("\nCurrent learning rate: %f" % cur_lr)
    trainer.set_learning_rate(cur_lr)
    train_data = ContextLoader(vocab, batch_size = 32, folder_name = 'data/train_full/', ctx = ctx)
    for pre_contexts, post_contexts, in_tokens, target_tokens, target_mask in train_data.iterator():
        with mx.autograd.record():
            output_tokens = model(pre_contexts, post_contexts, in_tokens)
            L = loss(output_tokens, target_tokens) # TODO(Ben): zero out the irrelevant padding tokens
            L_masked = L * target_mask
            L_masked.backward()
        grads = [i.grad(ctx) for i in model.collect_params().values()]
        mx.gluon.utils.clip_global_norm(grads, options.clip * data_options['context_width'] * options.batch_size) # TODO(Ben): adjust clipping for type of network
        trainer.step(options.batch_size)
        
        if iteration % 100 == 0:
            validation_losses.append(validation_loss())
        if iteration % 500 == 0:
            print("%dth iteration. Saving." % iteration)
            model.collect_params().save(save_dir + 'epoch-%d-%.6d.params' % (epoch, len(validation_losses)))
            validation_loss(num_minibatches = 1, verbose = True)
            np.save(save_dir + 'validation_losses', np.array(validation_losses))
        if time.time() - start_global > options.time_limit:
            print("Time limit reached. Ending epoch.")
            break
        iteration += 1
    print("Epoch completed. %d iterations" % iteration)
    if time.time() - start_global > options.time_limit:
        break
