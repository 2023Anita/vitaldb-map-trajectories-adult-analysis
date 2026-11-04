import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if len(sys.argv) != 3:
    raise SystemExit('Usage: gap_episode_sensitivity.py BUNDLE_DIR EMPTY_OUTPUT_DIR')
bundle = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
if not out.is_dir() or any(out.iterdir()):
    raise SystemExit('Output directory must already exist and be empty')
target = out / 'gap_episode_sensitivity_frame.csv'
summary_path = out / 'gap_episode_sensitivity_summary.csv'
if target.exists() or summary_path.exists():
    raise FileExistsError('Sensitivity output already exists')
script = bundle / 'code/s3_dynamic_hypotension_analysis.py'
spec = importlib.util.spec_from_file_location('gap_s3', script)
s3 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = s3
spec.loader.exec_module(s3)
selection = pd.read_csv(bundle / 'data/s3_selected_map_tracks.csv').set_index('caseid')
frame = pd.read_csv(bundle / 'data/adult_phenotype_frame.csv')
assert len(frame) == 5501 and frame.caseid.is_unique and selection.index.is_unique
assert frame.age.notna().all() and np.isfinite(frame.age).all() and (frame.age >= 18).all()
cache = bundle / 'data/map_tracks'

def split_gap_episode(tw, end, threshold=65):
    t = tw['time'].to_numpy(dtype=float)
    v = tw['map'].to_numpy(dtype=float)
    raw = np.r_[t[1:], end] - t
    valid = raw > 0
    v, raw = v[valid], raw[valid]
    capped = np.minimum(raw, s3.MAX_INTERVAL_SECONDS)
    n, current, longest = 0, 0., 0.
    for value, delta, interval in zip(v, raw, capped):
        if value < threshold:
            if current == 0:
                n += 1
            current += interval
            longest = max(longest, current)
        else:
            current = 0.
        if delta > s3.MAX_INTERVAL_SECONDS:
            current = 0.
    return n, longest / 60.

counts, durations = [], []
for i, row in enumerate(frame.itertuples(index=False), 1):
    tid = selection.loc[row.caseid, 'tid']
    track = s3.read_track(cache / f'{tid}.csv.gz')
    tw = s3.window_track(track, float(row.opstart), float(row.opend))
    n, duration = split_gap_episode(tw, float(row.opend))
    counts.append(n)
    durations.append(duration)
    if i % 500 == 0:
        print(f'processed {i}/{len(frame)}', flush=True)

alternative = frame.copy()
alternative['episode_count_lt_65'] = counts
alternative['longest_episode_min_lt_65'] = durations
complete, _, _ = s3.derive_phenotypes(alternative)
assert len(complete) == len(frame) and complete.caseid.tolist() == frame.caseid.tolist()
original = frame['trajectory_cluster'].to_numpy()
new = complete['trajectory_cluster'].to_numpy()
old_levels, new_levels = sorted(set(original)), sorted(set(new))
assert len(old_levels) == len(new_levels) == 4
best = max(itertools.permutations(old_levels), key=lambda p: sum(np.sum((new == c) & (original == v)) for c, v in zip(new_levels, p)))
alignment = dict(zip(new_levels, best))
aligned = np.array([alignment[c] for c in new])
result = pd.DataFrame({'caseid':frame.caseid, 'subjectid':frame.subjectid,
    'original_episode_count':frame.episode_count_lt_65, 'split_episode_count':counts,
    'original_longest_min':frame.longest_episode_min_lt_65, 'split_longest_min':durations,
    'original_cluster':original, 'new_cluster':new, 'new_cluster_aligned':aligned})
result.to_csv(target, index=False)
pd.DataFrame([{'cases':len(frame), 'count_changed':int(np.sum(frame.episode_count_lt_65.to_numpy()!=np.array(counts))),
    'longest_changed':int(np.sum(~np.isclose(frame.longest_episode_min_lt_65.to_numpy(),durations))),
    'aligned_label_changed':int(np.sum(original!=aligned)), 'note':'fixed adult cohort; gap >300 seconds resets episode; no change to coverage'}]).to_csv(summary_path,index=False)
print(pd.read_csv(summary_path).to_string(index=False))
