## General
* Tokens are split into subtokens so that it's agnostic to whether you use `camelCase` or `c_style` naming conventions. It's also invariant to case after splitting into subtokens. This allows you to unify the tokens with either style, but it does lose _some_ information.
* When loading in contexts, we sample by size of the file. This is proportional to the number of contexts in the file, which will oversample cases where there are a high number of contexts. I'm still not sure if this is preferable.
* `<<start_id>>` and `<<end_id>>` are used to guide seq2seq, but we also need `<<end_id>>` to distinguish `ClassName varName` from `classNameVarName`.

## On Processing
* For the future, make sure lone underscores appear as their own characters.
* The current scheme makes something like `_0` appear to be an underscore followed by an integer literal.
* We don't include predicting `<unk>` tokens, since they make up only around 3-4% of the dataset and it would not make the network output any useful information. 
