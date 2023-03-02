import pandas as pd
import numpy as np

ds = pd.read_csv('../tract_merged.csv')

rent_twobed = 0
ln_wage_growth = 0
ann_avg_job_growth = 0
gsmn_math = 0
emp2000 = 0
jobs_total = 0
jobs_highpay = 0
job_density = 0

for index, row in ds.iterrows():
    if np.isnan(row['rent_twobed2015']):
        rent_twobed += 1
    if np.isnan(row['ln_wage_growth_hs_grad']):
        ln_wage_growth += 1
    if np.isnan(row['ann_avg_job_growth_2004_2013']):
        ann_avg_job_growth += 1
    if np.isnan(row['gsmn_math_g3_2013']):
        gsmn_math += 1
    if np.isnan(row['emp2000']):
        emp2000 += 1
    if np.isnan(row['jobs_total_5mi_2015']):
        jobs_total += 1
    if np.isnan(row['jobs_highpay_5mi_2015']):
        jobs_highpay += 1
    if np.isnan(row['job_density_2013']):
        job_density += 1

print(f"RTB: {rent_twobed}")
print(f"LWG: {ln_wage_growth}")
print(f"AAJB: {ann_avg_job_growth}")
print(f"MATH: {gsmn_math}")
print(f"EMP: {emp2000}")
print(f"JTOTAL: {jobs_total}")
print(f"JHIGH: {jobs_highpay}")
print(f"JDEN: {job_density}")
