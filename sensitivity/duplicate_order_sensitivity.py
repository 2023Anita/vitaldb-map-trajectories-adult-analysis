import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd

if len(sys.argv) != 4:
    raise SystemExit('Usage: duplicate_order_sensitivity.py BUNDLE_DIR QA_CSV EMPTY_OUTPUT_DIR')
bundle=Path(sys.argv[1]).resolve(); qa_path=Path(sys.argv[2]).resolve(); out_dir=Path(sys.argv[3]).resolve()
if not out_dir.is_dir() or any(out_dir.iterdir()):
    raise SystemExit('Output directory must already exist and be empty')
spec=importlib.util.spec_from_file_location('adult_s3_dup',bundle/'code/s3_dynamic_hypotension_analysis.py')
s3=importlib.util.module_from_spec(spec);sys.modules[spec.name]=s3;spec.loader.exec_module(s3)
data=pd.read_csv(bundle/'data/adult_phenotype_frame.csv')
assert data.age.notna().all() and np.isfinite(data.age).all() and (data.age >= 18).all()
sel=pd.read_csv(bundle/'data/s3_selected_map_tracks.csv')
qa=pd.read_csv(qa_path)
ids=set(qa.loc[qa.conflicting_duplicate_timestamps>0,'caseid'])
work=data.copy()
features=['mean_map','min_map','sd_map','auc_per_hour_lt_65','pct_time_lt_65','early_map','middle_map','late_map','slope_norm','recovery_from_nadir','episode_count_lt_65','longest_episode_min_lt_65']
for index,row in work[work.caseid.isin(ids)].iterrows():
    tid=sel.loc[sel.caseid==row.caseid,'tid'].iloc[0]
    raw=pd.read_csv(bundle/'data/map_tracks'/f'{tid}.csv.gz',compression='gzip').iloc[:,:2]
    raw.columns=['time','map'];raw=raw.apply(pd.to_numeric,errors='coerce')
    stable=raw[(raw['map']>=s3.PHYS_MIN)&(raw['map']<=s3.PHYS_MAX)].sort_values('time',kind='stable').reset_index(drop=True)
    re=s3.trajectory_features(row,stable)
    for col in features: work.at[index,col]=re[col]
base,_,_=s3.derive_phenotypes(data)
alt,_,_=s3.derive_phenotypes(work)
assert base.caseid.equals(alt.caseid) and base.caseid.equals(data.caseid)
def ari(a,b):
    counts=pd.crosstab(a,b).to_numpy(dtype=float); c2=lambda x:x*(x-1)/2
    total=c2(counts.sum());r=c2(counts.sum(axis=1)).sum();c=c2(counts.sum(axis=0)).sum();return (c2(counts).sum()-r*c/total)/((r+c)/2-r*c/total)
out=pd.DataFrame([{'adult_cases':len(data),'conflict_cases':int(work.caseid.isin(ids).sum()),'baseline_reproduces_original_labels':bool((base.trajectory_cluster.to_numpy()==data.trajectory_cluster.to_numpy()).all()),'cluster_ari':ari(base.trajectory_cluster,alt.trajectory_cluster),'same_named_phenotype':int((base.trajectory_phenotype.to_numpy()==alt.trajectory_phenotype.to_numpy()).sum())}])
out.to_csv(out_dir/'duplicate_order_sensitivity.csv',index=False)
print(out.to_string(index=False))
