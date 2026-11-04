import csv
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
source = root / 'adult_phenotype_frame.csv'
output = root / 'table1_complete_working.csv'
if output.exists():
    raise FileExistsError(output)
with source.open(encoding='utf-8', newline='') as f:
    data = list(csv.DictReader(f))

groups = [
    ('Overall', None),
    ('Preserved high MAP', 'preserved_high_map'),
    ('Moderate or variable', 'moderate_low_burden_or_variable_map'),
    ('Early low with recovery', 'early_low_map_with_recovery'),
    ('Sustained severe', 'sustained_severe_hypotension_burden'),
]
assert len(data) == 5501 and len({r['caseid'] for r in data}) == 5501
assert all(float(r['age']) >= 18 for r in data)
assert sum(r['icu_use'] == '1' for r in data) == 1149
assert sum(1 for r in data if r['trajectory_phenotype'] not in {g[1] for g in groups}) == 0
subsets = [[r for r in data if key is None or r['trajectory_phenotype'] == key] for _, key in groups]
assert [len(s) for s in subsets] == [5501, 1597, 2386, 1029, 489]

def numeric(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None

def continuous(rows, field):
    values = [v for r in rows if (v := numeric(r[field])) is not None]
    missing = len(rows) - len(values)
    if not values:
        return f'NA; missing {missing}'
    if len(values) == 1:
        q1 = q3 = values[0]
    else:
        q1, _, q3 = statistics.quantiles(values, n=4, method='inclusive')
    return f'{statistics.median(values):.1f} ({q1:.1f}-{q3:.1f}); missing {missing}'

def categorical(rows, field, value):
    n = sum(r[field] == value for r in rows)
    return f'{n} ({100*n/len(rows):.1f}%)'

result = [['Variable / category'] + [f'{name} (n={len(rows)})' for (name, _), rows in zip(groups, subsets)]]
for label, field in [
    ('Age, years; median (IQR)', 'age'), ('BMI; median (IQR)', 'bmi'),
    ('Operative duration, min; median (IQR)', 'op_duration_min'),
    ('Preoperative hemoglobin; median (IQR)', 'preop_hb'),
    ('Preoperative creatinine; median (IQR)', 'preop_cr'),
    ('Preoperative albumin; median (IQR)', 'preop_alb'),
]:
    result.append([label] + [continuous(rows, field) for rows in subsets])

for label, field, levels in [
    ('Sex', 'sex', [('Male', 'M'), ('Female', 'F'), ('Missing', '')]),
    ('ASA, recorded value', 'asa', [('1', '1.0'), ('2', '2.0'), ('3', '3.0'), ('4', '4.0'), ('6 (verify coding)', '6.0'), ('Missing', '')]),
    ('Emergency operation', 'emop', [('Yes', '1'), ('No', '0'), ('Missing', '')]),
    ('Surgical department', 'department', [('General surgery', 'General surgery'), ('Thoracic surgery', 'Thoracic surgery'), ('Gynecology', 'Gynecology'), ('Urology', 'Urology'), ('Missing', '')]),
    ('MAP source', 'map_source_type', [('Invasive', 'invasive'), ('Noninvasive', 'nibp'), ('Missing', '')]),
    ('Recorded ICU use', 'icu_use', [('Yes', '1'), ('No', '0'), ('Missing', '')]),
]:
    allowed = {value for _, value in levels}
    observed = {r[field] for r in data}
    if not observed <= allowed:
        raise ValueError(f'{field}: unexpected categories {observed - allowed}')
    for category, value in levels:
        result.append([f'{label}: {category}'] + [categorical(rows, field, value) for rows in subsets])

with output.open('x', encoding='utf-8', newline='') as f:
    csv.writer(f).writerows(result)
print(f'{output}: {len(result)-1} rows; denominators and cohort counts verified')
