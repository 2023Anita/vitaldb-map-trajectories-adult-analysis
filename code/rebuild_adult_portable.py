import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description='Rebuild the adult feature frame from the included selected-track cache.')
parser.add_argument('--data-dir', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
data_dir = args.data_dir.resolve()
output_dir = args.output_dir.resolve()
if not output_dir.is_dir() or any(output_dir.iterdir()):
    raise SystemExit('Output directory must already exist and be empty')

spec = importlib.util.spec_from_file_location('bundle_s3', Path(__file__).with_name('s3_dynamic_hypotension_analysis.py'))
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.CACHE = data_dir / 'map_tracks'
cases = pd.read_csv(data_dir / 'cases.csv')
cases = cases.rename(columns={cases.columns[0]: 'caseid'})
selection = pd.read_csv(data_dir / 's3_selected_map_tracks.csv')
if not cases.caseid.is_unique or not selection.caseid.is_unique:
    raise ValueError('Duplicate case identifiers')
adult = cases[pd.to_numeric(cases.age, errors='coerce') >= 18].copy()
known = adult[pd.to_numeric(adult.icu_days, errors='coerce').notna()].copy()
features = module.build_feature_table(known, selection)
frame = module.prepare_analysis_frame(known, features)
eligible = module.analysis_cohort(frame)
complete, selection_k, profiles = module.derive_phenotypes(eligible)
if not (complete.caseid.is_unique and complete.age.notna().all() and (complete.age >= 18).all() and complete.icu_days.notna().all()):
    raise ValueError('Adult analysis cohort invariants failed')
for name, table in [('adult_features.csv', features), ('adult_eligible.csv', eligible),
                    ('adult_phenotype_frame.csv', complete), ('adult_cluster_selection.csv', selection_k),
                    ('adult_cluster_profiles.csv', profiles)]:
    table.to_csv(output_dir / name, index=False)
print(f'adult={len(adult)} eligible={len(eligible)} complete={len(complete)} subjects={complete.subjectid.nunique()} events={complete.icu_use.sum()}')
