"""A26 reconcile: compare report v2 cost table (reports/tables/sacct.txt) with pass B (train_costs.json).
Stdlib only. Usage: python3 costs_reconcile.py REPO_ROOT OUT_JSON"""
import json, re, sys, collections
root, out = sys.argv[1], sys.argv[2]
def tres(s, key):
    m = re.search(r'(?:^|,)' + key + r'=(\d+)', s)
    return int(m.group(1)) if m else 0
rep = []
for line in open(f'{root}/reports/tables/sacct.txt'):
    f = line.rstrip('\n').split('|')
    if len(f) < 6: continue
    jid, name, state, part, el, at = f[:6]
    rep.append(dict(jobid=jid.split('_')[0].split('.')[0], name=name, state=state, elapsed_s=int(el or 0),
                    gpus=tres(at, 'gres/gpu'), cpus=tres(at, 'cpu')))
pb = json.load(open(f'{root}/research/audit-a26/pass-b/data/train_costs.json'))
def agg(rows):
    d = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for r in rows:
        d[r['jobid']][0] += r['elapsed_s'] * r['gpus'] / 3600
        d[r['jobid']][1] += r['elapsed_s'] * r['cpus'] / 3600
        d[r['jobid']][2] += 1
    return d
A, B = agg(rep), agg(pb['rows'])
res = dict(report_jobs=len(A), report_allocs=len(rep), report_gpu_h=sum(v[0] for v in A.values()),
           report_cpu_h=sum(v[1] for v in A.values()), passb_jobs=len(B), passb_allocs=len(pb['rows']),
           passb_gpu_h=sum(v[0] for v in B.values()), passb_cpu_h=sum(v[1] for v in B.values()))
expid = {r['jobid']: r.get('expid') for r in pb['rows']}
only_b = sorted(set(B) - set(A)); only_a = sorted(set(A) - set(B))
res['only_in_passb'] = [dict(jobid=j, expid=expid.get(j), gpu_h=B[j][0], cpu_h=B[j][1], allocs=B[j][2],
                             states=sorted({r['state'] for r in pb['rows'] if r['jobid'] == j})) for j in only_b]
res['only_in_report'] = only_a
res['only_b_gpu_h'] = sum(B[j][0] for j in only_b); res['only_b_cpu_h'] = sum(B[j][1] for j in only_b)
common = set(A) & set(B)
diffs = [(j, B[j][0] - A[j][0], B[j][1] - A[j][1], B[j][2] - A[j][2]) for j in common
         if abs(B[j][0] - A[j][0]) > 1e-9 or abs(B[j][1] - A[j][1]) > 1e-9 or B[j][2] != A[j][2]]
res['common_jobs'] = len(common)
res['common_diffs'] = [dict(jobid=j, expid=expid.get(j), d_gpu_h=g, d_cpu_h=c, d_allocs=n) for j, g, c, n in diffs]
res['common_gpu_h_report'] = sum(A[j][0] for j in common); res['common_gpu_h_passb'] = sum(B[j][0] for j in common)
res['common_cpu_h_report'] = sum(A[j][1] for j in common); res['common_cpu_h_passb'] = sum(B[j][1] for j in common)
res['passb_running_rows'] = pb['running_rows']
json.dump(res, open(out, 'w'), indent=1)
for k, v in res.items():
    if not isinstance(v, list): print(k, v)
for x in res['only_in_passb']: print('onlyB', x)
print('only report', only_a)
for x in res['common_diffs']: print('diff', x)
