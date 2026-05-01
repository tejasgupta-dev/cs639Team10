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


def read_csv_or_none(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f'[skipped] {path.relative_to(REPO_ROOT)} (not found)')
        return None


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
        if label == 'Depth experiment':
            note = '(different cluster config — values not comparable to q1000/math_l5)'
            text.append('')
            text.append(note)
            md.append('')
            md.append(f'_{note}_')

    print('\n'.join(text))
    OUT_MD.write_text('\n'.join(md) + '\n')
    print(f'\nWrote {OUT_MD.relative_to(REPO_ROOT)}')


if __name__ == '__main__':
    main()
