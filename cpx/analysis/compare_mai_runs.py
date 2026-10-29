"""Report MAI paired success/call metrics without selecting best attempts."""
import json
import statistics
from pathlib import Path

TAGS = {
    'BASE': 'mai8b_base_full116_calls50_12w_20261026',
    'FULL45（修复后）': 'mai8b_full45_full116_calls50_recoveryfix_20261026',
    'FULL45（修复前）': 'mai8b_full45_full116_calls50_emptyshared_20261025',
}


def valid(row):
    return row.get('rc') == 0 and not row.get('error') and row.get('success') is not None


def mean_calls(rows):
    calls = [r['model_calls'] for r in rows if valid(r) and r.get('model_calls') is not None]
    return statistics.mean(calls) if calls else None


def compare(keyed):
    base = keyed['BASE']
    assert len(base) == 116
    output = []
    for label, rows in keyed.items():
        assert set(rows) == set(base), label
        good = [r for r in rows.values() if valid(r)]
        success = [r for r in good if r['success'] == 1]
        common = [t for t in base if valid(base[t]) and base[t]['success'] == 1
                  and valid(rows[t]) and rows[t]['success'] == 1]
        row = dict(method=label, tasks=len(rows), valid_tasks=len(good),
                   successes=len(success), success_rate=len(success)/len(rows),
                   partial_tasks=[r['task'] for r in good if 0 < r['success'] < 1],
                   mean_calls=mean_calls(good), success_mean_calls=mean_calls(success),
                   missing_call_tasks=[r['task'] for r in rows.values() if r.get('model_calls') is None],
                   errors=[dict(task=r['task'], error=r.get('error')) for r in rows.values() if not valid(r)])
        if label != 'BASE':
            cm = mean_calls([rows[t] for t in common])
            bm = mean_calls([base[t] for t in common])
            row.update(common_success_tasks=common, common_count=len(common),
                       common_mean_calls=cm, common_base_mean_calls=bm,
                       common_change=cm/bm-1 if bm else None,
                       all_mean_change=row['mean_calls']/mean_calls(base.values())-1)
        output.append(row)
    return output


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    keyed={}
    for label, tag in TAGS.items():
        rows=[json.loads(s) for s in (args.root/tag/'runs.jsonl').read_text().splitlines() if s.strip()]
        assert len(rows)==116 and len({r['task'] for r in rows})==116, tag
        keyed[label]={r['task']:r for r in rows}
    result=compare(keyed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
