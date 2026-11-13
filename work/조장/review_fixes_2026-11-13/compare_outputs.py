"""Compare code-cell outputs of two notebooks (baseline vs re-executed).

Usage: python compare_outputs.py <baseline.ipynb> <new.ipynb> [--swap01]
--swap01: the new notebook has cells 0 and 1 swapped relative to the baseline (both markdown).
Prints, per differing code cell, a short unified diff of the text outputs and whether images changed.
"""
import difflib
import io
import json
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
base = json.load(open(sys.argv[1], encoding='utf-8'))['cells']
new = json.load(open(sys.argv[2], encoding='utf-8'))['cells']
assert len(base) == len(new), (len(base), len(new))


def text_of(cell):
    parts, images, errors = [], 0, []
    for o in cell.get('outputs', []):
        if o['output_type'] == 'stream':
            parts.append(''.join(o['text']))
        elif o['output_type'] == 'error':
            errors.append(o.get('ename', '?') + ': ' + str(o.get('evalue', ''))[:200])
        else:
            data = o.get('data', {})
            if 'text/html' in data:
                parts.append(re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', ''.join(data['text/html']))))
            elif 'text/plain' in data:
                parts.append(''.join(data['text/plain']))
            images += 'image/png' in data
    return '\n'.join(parts), images, errors


same = differ = 0
for i, (b, n) in enumerate(zip(base, new)):
    if b['cell_type'] != n['cell_type']:
        print(f'CELL {i}: type differs {b["cell_type"]} vs {n["cell_type"]}')
        continue
    if b['cell_type'] != 'code':
        continue
    bt, bi, be = text_of(b)
    nt, ni, ne = text_of(n)
    if ne:
        print(f'CELL {i}: ERROR OUTPUT {ne}')
    if bt == nt and bi == ni:
        same += 1
        continue
    differ += 1
    first = ''.join(n['source']).split('\n')[0][:80]
    print(f'===== CELL {i} differs (images {bi} -> {ni}, chars {len(bt)} -> {len(nt)}) | {first}')
    diff = list(difflib.unified_diff(bt.splitlines(), nt.splitlines(), lineterm='', n=0))
    for line in diff[:40]:
        print('   ', line[:260])
    if len(diff) > 40:
        print(f'    ... {len(diff) - 40} more diff lines')
print(f'code cells same: {same}, differ: {differ}')
counts = [c.get('execution_count') for c in new if c['cell_type'] == 'code']
print('execution_count sequential from 1:', counts == list(range(1, len(counts) + 1)))
