import pandas as pd, numpy as np, os
p = 'acoustic/0Nm_BPFI_03.csv'   # change to one file path
print('exists?', os.path.exists(p))
df = pd.read_csv(p, nrows=5)
print('columns', df.columns.tolist())
time_col = [c for c in df.columns if 'time' in c.lower() and 'stamp' in c.lower()]
time_col = time_col[0] if time_col else df.columns[0]
print('time_col used:', time_col)
head = pd.read_csv(p, nrows=100)[time_col].to_numpy()
print('first 10 times:', head[:10])
print('diffs:', np.diff(head[:20]))
print('are diffs >0?', np.any(np.diff(head[:20])>0))
