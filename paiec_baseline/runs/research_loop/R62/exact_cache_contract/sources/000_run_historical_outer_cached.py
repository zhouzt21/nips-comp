"""Run the frozen scorer with an independently audited bit-exact encoding cache."""
from historical_positive_common import *
from historical_exact_encoding_cache import ExactEncodingCache
import historical_positive_episodes
import run_historical_positive_outer

if __name__=='__main__':
    assert read(ROOT/'R62/encoding_equivalence.json')['bitwise_equal']
    freeze(ROOT/'R62/exact_cache_contract',[Path(__file__),REPO/'research_loop/historical_exact_encoding_cache.py',
        ROOT/'R62/encoding_equivalence.json',ROOT/'R62/evaluation_contract/contract.json'],
        {'reason':'full unchanged scorer cost; bit-exact cache only, no predictor or selection change'})
    cache=ExactEncodingCache()
    historical_positive_episodes.encode_mode=cache
    run_historical_positive_outer.main('R62')
    write_json(ROOT/'R62/exact_cache_stats.json',cache.stats())
    check_freeze(ROOT/'R62/exact_cache_contract')
