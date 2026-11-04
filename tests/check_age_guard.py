import csv
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
test_dir = root / 'tests' / 'generated'
test_dir.mkdir(exist_ok=False)
features = ['mean_map', 'min_map', 'sd_map', 'auc_per_hour_lt_65', 'pct_time_lt_65',
            'early_map', 'middle_map', 'late_map', 'slope_norm', 'recovery_from_nadir',
            'episode_count_lt_65', 'longest_episode_min_lt_65']
base = {'caseid': '1', 'subjectid': '1', 'icu_days': '0', 'icu_use': '0',
        'trajectory_cluster': '1', **{key: '1' for key in features}}
cases = {'minor': '17', 'missing': '', 'nonnumeric': 'bad', 'infinite': 'Inf', 'missing_column': None}
scripts = ['reanalysis_or.R', 'internal_validation.R', 'sensitivity.R', 'cluster_uncertainty.R']
for name, age in cases.items():
    row = dict(base)
    if age is not None:
        row['age'] = age
    path = test_dir / f'{name}.csv'
    with path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    for script in scripts:
        rscript = sys.argv[1] if len(sys.argv) > 1 else 'Rscript'
        result = subprocess.run([rscript, str(root / 'code' / script), str(path), str(test_dir)],
                                capture_output=True, text=True, timeout=30)
        if result.returncode == 0 or 'Adult cohort requires finite age >= 18' not in result.stderr:
            raise AssertionError(f'{script} did not reject {name}: {result.stderr}')
        print(f'PASS {script} rejects {name}')
print('20 negative age-validation checks passed; full-cohort positive tests are separate.')
