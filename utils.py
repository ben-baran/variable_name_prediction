import re

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
