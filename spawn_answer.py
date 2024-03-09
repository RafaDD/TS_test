import numpy as np
import pandas as pd

result = []
pred = np.load('result/Paris.npy').T
index = np.load('datasets/Paris/test_ids.npy', allow_pickle=True).item()
keys = np.sort(list(index.keys()))
i = 0
for k in keys:
    times = index[k]
    result += list(pred[i, [times]].reshape(-1))
    i += 1

result = np.around(np.array(result), 1)
ids = np.arange(1, len(result)+1)
df = pd.DataFrame(columns=['id', 'estimate_q'])
df['id'] = ids
df['estimate_q'] = result
df.to_csv('result/submit.csv', index=False)