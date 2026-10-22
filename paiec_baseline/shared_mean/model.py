"""Research predictor using only the same-benchmark evidence actually supplied."""
PEER_STRENGTH=8.


def from_counts(own_successes,own_count,peer_successes,peer_count,kappa=PEER_STRENGTH):
    prior=(kappa*.5+peer_successes)/(kappa+peer_count)
    return float((2*prior+own_successes)/(2+own_count))


def predict(input,labeled=None):
    subject,item=input
    if not labeled:return .5
    benchmark=item.get("benchmark_id")
    if not isinstance(benchmark,str) or not benchmark:raise ValueError("benchmark_id required")
    own_s=own_n=peer_s=peer_n=0
    for (other_subject,other_item),label in labeled:
        if other_item.get("benchmark_id")!=benchmark:continue
        if label not in (0,1):raise ValueError("binary label required")
        if other_subject==subject:own_s+=label;own_n+=1
        else:peer_s+=label;peer_n+=1
    return from_counts(own_s,own_n,peer_s,peer_n)
