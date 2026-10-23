"""Infer text invariants only from the current allowed support inputs."""


def window(tokens):
    return tuple(tokens if len(tokens)<=126 else [*tokens[:96],*tokens[-30:]])


def template(support):
    seq=[tuple(x) for x in support]
    if len(seq)<3 or len(set(seq))<2:return None
    shortest=min(map(len,seq));p=0
    while p<shortest and all(s[p]==seq[0][p] for s in seq):p+=1
    z=0
    while z<shortest-p and all(s[-1-z]==seq[0][-1-z] for s in seq):z+=1
    if p+z<32:return None
    return seq[0][:p],seq[0][-z:] if z else ()


def retained(tokens,rule):
    t=tuple(tokens)
    if rule is None:return window(t)
    prefix,suffix=rule;p=len(prefix);z=len(suffix)
    if len(t)<p+z or t[:p]!=prefix or (z and t[-z:]!=suffix):return window(t)
    left=max(p-8,0);right=len(t)-max(z-8,0)
    return window(t[left:right])
