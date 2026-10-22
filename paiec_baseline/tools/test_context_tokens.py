from mlp.context_tokens import template,retained,window


def test_support_affixes_expose_variable_content():
    prefix=tuple(range(60));suffix=tuple(range(200,240))
    support=[prefix+(1000+i,)+suffix for i in range(3)]
    rule=template(support)
    assert rule==(prefix,suffix)
    for i,s in enumerate(support):assert retained(s,rule)==prefix[-8:]+(1000+i,)+suffix[:8]
    assert retained(prefix+(9999,)+suffix,rule)==prefix[-8:]+(9999,)+suffix[:8]


def test_mismatched_target_and_prefix_budget_do_not_change_rule():
    support=[tuple(range(70))+(100+i,) for i in range(7)];rule=template(support[:3])
    bad=(999,)+support[0][1:]
    assert retained(bad,rule)==window(bad)
    assert template(support[:1]) is None
    retained(bad,rule)
    assert template(support[:3])==rule
    assert template(list(reversed(support[:3])))==rule


def test_constant_and_short_shared_context_fallback():
    assert template([(1,2,3)]*31) is None
    assert template([(1,2,i) for i in range(3)]) is None
    long=tuple(range(400))
    assert len(window(long))==126 and window(long)==long[:96]+long[-30:]
