"""Consolidate strategy_fall result CSVs into a single readable summary.

Reads existing CSVs only. Prints a text summary to stdout and writes the
same content as evaluation_summary.md at the repo root.
"""

from datetime import datetime
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / 'strategy_fall' / 'results'
OUT_MD = REPO_ROOT / 'evaluation_summary.md'

STRUCTURAL = [
    ('GSM8K (q1000)',    RESULTS_DIR / 'q1000'   / 'strategy_collapse_report_q1000.csv'),
    ('MATH Level 5',     RESULTS_DIR / 'math_l5' / 'strategy_collapse_report_math_l5.csv'),
    ('Depth experiment', RESULTS_DIR / 'depth'   / 'strategy_collapse_report_depth.csv'),
]

CAUSAL_CONDITIONS = [
    ('GSM8K, RL',    RESULTS_DIR / 'causal'),
    ('GSM8K, SFT',   RESULTS_DIR / 'causal_sft'),
    ('MATH L5, RL',  RESULTS_DIR / 'causal_math_l5'),
    ('MATH L5, SFT', RESULTS_DIR / 'causal_math_l5_sft'),
]

CAUSAL_CAVEATS = {
    'GSM8K, SFT':   "SFT-GSM8K outputs use 'Final Answer:' without '####' / '\\boxed{}'; absolute accuracies reflect parser limitation, drops still informative.",
}

LEGACY_CSVS = [
    (RESULTS_DIR / 'strategy_collapse_report.csv',
     'older 4-col schema; superseded by q1000/strategy_collapse_report_q1000.csv'),
    (RESULTS_DIR / 'Qwen2.5-7B-Instruct-AWQ_details.csv',
     '50-row precursor; superseded by q1000/Qwen2.5-7B-Instruct-AWQ_traces-q1000_details.csv'),
    (RESULTS_DIR / 'DeepSeek-R1-Distill-Qwen-7B-Floppanacci-AWQ_details.csv',
     '50-row precursor; superseded by q1000/DeepSeek-R1-Distill-Qwen-7B-Floppanacci-AWQ_traces-q1000_details.csv'),
    (RESULTS_DIR / 'qwen2.5-7b-awq-4bit_details.csv',
     '50-row precursor with empty migration cols; superseded by q1000/qwen2.5-7b-awq-4bit_traces-q1000_details.csv'),
]


def read_csv_or_none(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f'[skipped] {path.relative_to(REPO_ROOT)} (not found)')
        return None


def deltas_pairwise(df, metric_cols):
    roles = {}
    for _, row in df.iterrows():
        n = str(row['model']).lower()
        if 'deepseek' in n or 'floppanacci' in n:
            roles['RL'] = row
        elif '4bit' in n:
            roles['base'] = row
        elif 'instruct' in n:
            roles['SFT'] = row
    out = []
    for a, b in [('RL', 'SFT'), ('RL', 'base'), ('SFT', 'base')]:
        if a not in roles or b not in roles:
            continue
        parts = [f'{a} - {b}:']
        for col in metric_cols:
            d = float(roles[a][col]) - float(roles[b][col])
            sign = '+' if d >= 0 else '-'
            parts.append(f'{col} {sign}{abs(d):.4f}')
        out.append('  '.join(parts))
    return out


def causal_block(directory):
    summary = read_csv_or_none(directory / 'causal_summary.csv')
    details = read_csv_or_none(directory / 'causal_details.csv')
    if summary is None or details is None:
        return None
    tags = [c for c in summary.columns if c != 'type']
    ctrl = summary[summary['type'] == 'control'].iloc[0]
    intv = summary[summary['type'] == 'intervention'].iloc[0]
    rows = []
    for t in tags:
        c = float(ctrl[t])
        i = float(intv[t])
        rows.append({'tag': t, 'control': round(c, 4), 'intervention': round(i, 4), 'drop': round(c - i, 4)})
    return pd.DataFrame(rows), int(details['qid'].nunique())


def df_to_md(df):
    headers = list(df.columns)
    lines = ['| ' + ' | '.join(str(h) for h in headers) + ' |']
    lines.append('| ' + ' | '.join('---' for _ in headers) + ' |')
    for row in df.values.tolist():
        lines.append('| ' + ' | '.join(str(v) for v in row) + ' |')
    return '\n'.join(lines)


def main():
    text = []
    md = []

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    text.append('=' * 64)
    text.append('EVALUATION SUMMARY')
    text.append(f'Generated: {timestamp}')
    text.append('=' * 64)
    md.append('# Evaluation Summary')
    md.append('')
    md.append(f'_Generated: {timestamp}_')

    for label, path in STRUCTURAL:
        text.append('')
        text.append('-' * 64)
        text.append(f'Structural metrics — {label}')
        text.append('-' * 64)
        md.append('')
        md.append(f'## Structural metrics — {label}')
        md.append('')
        df = read_csv_or_none(path)
        if df is None:
            text.append('(no data)')
            md.append('_(no data)_')
            continue
        text.append(df.to_string(index=False))
        md.append(df_to_md(df))
        metric_cols = [c for c in df.columns if c != 'model']
        delta_lines = deltas_pairwise(df, metric_cols)
        if delta_lines:
            text.append('')
            text.append('Deltas:')
            md.append('')
            md.append('**Deltas:**')
            md.append('')
            for line in delta_lines:
                text.append('  ' + line)
                md.append(f'- {line}')
        if label == 'Depth experiment':
            note = '(different cluster config — values not comparable to q1000/math_l5)'
            text.append('')
            text.append(note)
            md.append('')
            md.append(f'_{note}_')

    for label, directory in CAUSAL_CONDITIONS:
        text.append('')
        text.append('-' * 64)
        text.append(f'Causal — {label}')
        text.append('-' * 64)
        md.append('')
        md.append(f'## Causal — {label}')
        md.append('')
        result = causal_block(directory)
        if result is None:
            text.append('(no data)')
            md.append('_(no data)_')
            continue
        table, n_questions = result
        text.append(table.to_string(index=False))
        md.append(df_to_md(table))
        text.append(f'n_questions = {n_questions}')
        md.append('')
        md.append(f'_n_questions = {n_questions}_')
        caveat = CAUSAL_CAVEATS.get(label)
        if caveat:
            text.append(f'note: {caveat}')
            md.append('')
            md.append(f'**Note:** {caveat}')

    text.append('')
    text.append('-' * 64)
    text.append('Files skipped (legacy / superseded)')
    text.append('-' * 64)
    md.append('')
    md.append('## Files skipped (legacy / superseded)')
    md.append('')
    for path, reason in LEGACY_CSVS:
        if not path.exists():
            continue
        rel = path.relative_to(REPO_ROOT)
        text.append(f'  {rel}  — {reason}')
        md.append(f'- `{rel}` — {reason}')

    print('\n'.join(text))
    OUT_MD.write_text('\n'.join(md) + '\n')
    print(f'\nWrote {OUT_MD.relative_to(REPO_ROOT)}')


if __name__ == '__main__':
    main()
