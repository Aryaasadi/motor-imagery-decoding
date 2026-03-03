from run_embc import run_subject
import pandas as pd

data_dir = r"D:\Articles\1-ongoing\2-IEEE\BCICIV2a"
k_list = [20, 40, 60, 80, 100, 120, 150, 200, 300, 400]

all_res = []
for subj in range(1, 10):
    r = run_subject(data_dir, subj, k_list, outer_splits=5, inner_splits=5, seed=0)
    all_res.append(r)

df = pd.DataFrame(all_res)
df.to_csv("embc_Tonly_results_per_subject.csv", index=False)
print(df)
