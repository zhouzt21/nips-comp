"""Require bitwise equality to every completed pre-cache prediction chunk."""
from historical_positive_common import *

def main():
    old=ROOT/'R62/incomplete_original/alignment_faking'
    new=ROOT/'R62/outer/fold2/alignment_faking'
    records=[];rows=0;values=0;unpaired=[]
    for path in sorted(old.glob('predictions_*.npz')):
        index=path.with_name(path.name.replace('predictions_','index_')).with_suffix('.json')
        if not index.exists():
            unpaired.append(path.name);continue
        assert read(index)==read(new/index.name),'Changed prediction identities'
        with np.load(path,allow_pickle=False) as a, np.load(new/path.name,allow_pickle=False) as b:
            assert a.files==b.files
            for k in a.files:
                assert np.array_equal(a[k],b[k]),(path.name,k,float(np.max(np.abs(a[k]-b[k]))))
                assert a[k].dtype==b[k].dtype and a[k].shape==b[k].shape
                assert a[k].tobytes()==b[k].tobytes(),(path.name,k,'byte mismatch')
                values+=a[k].size
            rows+=len(a['counts'])
        records.append({'file':path.name,'original_sha256':sha(path),'new_sha256':sha(new/path.name),
                        'all_arrays_bitwise_equal':True})
    assert records
    result={'complete_overlap_chunks':len(records),'query_variants':rows,'array_values_compared':values,
            'unpaired_interrupted_files':unpaired,'records':records,'max_absolute_error':0.,
            'checked_utc':now(),'comparison':'dtype, shape and raw bytes for every array',
            'scope':'prediction-preserving compute change; scientific contract unchanged'}
    write_json(ROOT/'R62/cache_overlap_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))

if __name__=='__main__':main()
