import random
import subprocess
import pprint
import time

def sample_hidden_per_side():
    p = random.random()
    return 64 * (2 ** random.randint(0, 5))

def sample_num_layers():
    return random.randint(1, 5)

def sample_embedding_size():
    return sample_hidden_per_side()

def sample_dropout():
    return random.uniform(0.25, 1.0)

def sample_clip():
    return random.uniform(0.1, 0.4)

def sample_lr():
    return random.triangular(3e-6, 2e-3, 3e-4)

def sample_lr_decay():
    return random.triangular(0.9, 1.0, 0.75)

def should_tie_weights():
    return random.random() > 0.9

learning_rates = [3e-4, 1e-4, 3e-5, 1e-5]#, 3e-6]
lr_decays = [1.0, 0.99, 0.98, 0.96, 0.94, 0.9, 0.85]#, 0.6]
ends = [0, 1, 1, 1, 4, 5, 6, 6]

while True:
    call = ['python']
    call += ['train.py']
    call += ['--epochs=1000'] # doesn't matter, we'll use time as the limit
    call += ['--batch-size=32'] # not optimized
    call += ['--hidden-per-side=%d' % sample_hidden_per_side()] 
    call += ['--num-layers=%d' % sample_num_layers()] 
    call += ['--embedding-size=%d' % sample_embedding_size()] 
    call += ['--dropout=%f' % sample_dropout()] 
    call += ['--clip=%f' % sample_clip()] 
    call += ['-lr=%f' % sample_lr()] 
    call += ['--lr-decay=%f' % sample_lr_decay()] 
    call += ['--time-limit=7200'] # two hours
    if should_tie_weights():
        call += ['--tie-weights']
    call += []
    
    pprint.pprint(call)
    subprocess.call(call)
    time.sleep(5) # this is so CUDA has time to clean up its memory