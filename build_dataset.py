# Griffin Davis, The University of Texas at Dallas
# (C) 2022
# Data source:
# Chetty, Raj; Friedman, John; Hendren, Nathaniel; Jones, Maggie R.; Porter, Sonya R., 2022, 
# "Replication Data for: The Opportunity Atlas: Mapping the Childhood Roots of Social Mobility", 
# https://doi.org/10.7910/DVN/NKCQM1, Harvard Dataverse, V1, UNF:6:wwWmCZy1LUqtq02qHdCKFQ== [fileUNF] 

import pandas as pd
import numpy as np

!wget -nc https://personal.utdallas.edu/~gcd/data/tract_outcomes_min.csv
outcomes = pd.read_csv('tract_outcomes_min.csv')
!wget -nc https://personal.utdallas.edu/~gcd/data/tract_covariates.csv
covariates = pd.read_csv('tract_covariates.csv')

outcomes['id'] = outcomes.apply(lambda row: str(row.state)+str(row.county)+str(row.tract), axis=1)
covariates['id'] = covariates.apply(lambda row: str(row.state)+str(row.county)+str(row.tract), axis=1)

cov_out = pd.merge(covariates, outcomes, on='id', how="outer", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)').sort_values(by=['id'])
cov_out = cov_out[cov_out['kfr_pooled_pooled_p1'].notna()].sort_values(['id'])
cov_out.to_csv('merged.csv', index=False)