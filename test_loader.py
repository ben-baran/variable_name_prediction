from utils import ContextLoader, ComboVocab, rough_print
import time

tic = time.time()
vocab = ComboVocab()
loader = ContextLoader(vocab, batch_size = 32, folder_name = 'data/train_tmp/')
toc = time.time()
print("Time for initial load:", toc - tic)
tic = toc

n_batches = 0
n_context_groups = 0
for pre_contexts, post_contexts, in_tokens, out_tokens, target_mask in loader.iterator():
    n_batches += 1
    n_context_groups += len(pre_contexts)
    
    for i in range(1):
        print('inputs:', vocab.to_tokens([int(x) for x in in_tokens[:, i].asnumpy()]))
        print('outputs:', vocab.to_tokens([int(x) for x in out_tokens.reshape((-1, 32))[:, i].asnumpy()]))
        indent_level = rough_print(vocab.to_tokens([int(x) for x in pre_contexts[i, :, 0].asnumpy()]))
        print("{{{{SLOT}}}}", end = '')
        rough_print(vocab.to_tokens([int(x) for x in post_contexts[i, :, 0].asnumpy()])[::-1], indent_level = indent_level, open_indent = False)
        print()
    if n_batches == 5:
        break
    
toc = time.time()
print("Total number of context groups:", n_context_groups)
print("Total batch load time: %f, number of batches: %d" % (toc - tic, n_batches))
print("Average time per batch:", (toc - tic) / n_batches)
