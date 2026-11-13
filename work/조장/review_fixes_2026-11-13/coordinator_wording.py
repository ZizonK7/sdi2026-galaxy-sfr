"""Coordinator wording corrections on top of the worker's edit (review items 3, 6, 7): make the sentences match the stored outputs exactly."""
import json, sys
path = sys.argv[1]
raw = open(path, 'rb').read()
crlf = b'\r\n' in raw
nb = json.loads(raw.decode('utf-8'))
fixes = [
    (113, '실제 SFR = 0인 은하(주황)는 0의 오른쪽(과대예측)에만 있다.', '실제 SFR = 0인 은하(주황)는 0이거나 그 오른쪽(과대예측)에만 있다.'),
    (120, '이웃 은하 피처는 0.002 이하로 작게 쓰였다', '이웃 은하 피처는 가장 큰 것도 0.002 수준으로 작게 쓰였다'),
    (120, '중심 은하는 어느 그룹에서든 0.4% 이하다', '중심 은하는 어느 그룹에서든 0.5% 미만이다'),
]
for idx, old, new in fixes:
    src = ''.join(nb['cells'][idx]['source'])
    assert src.count(old) == 1, (idx, old, src.count(old))
    nb['cells'][idx]['source'] = src.replace(old, new).splitlines(keepends=True)
text = json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=False) + '\n'
if crlf:
    text = text.replace('\n', '\r\n')
open(path, 'wb').write(text.encode('utf-8'))
print('applied', len(fixes))
