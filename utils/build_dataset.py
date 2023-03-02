# Griffin Davis, The University of Texas at Dallas
# (C) 2022
# Data source:
# Chetty, Raj; Friedman, John; Hendren, Nathaniel; Jones, Maggie R.; Porter, Sonya R., 2022, 
# "Replication Data for: The Opportunity Atlas: Mapping the Childhood Roots of Social Mobility", 
# https://doi.org/10.7910/DVN/NKCQM1, Harvard Dataverse, V1, UNF:6:wwWmCZy1LUqtq02qHdCKFQ== [fileUNF] 

import requests
import pandas as pd
import numpy as np

# Download data
outcomes_file = requests.get('https://personal.utdallas.edu/~gcd/data/tract_outcomes_min.csv', allow_redirects=True)
open('tract_outcomes_min.csv', 'wb').write(outcomes_file.content)
outcomes = pd.read_csv('tract_outcomes_min.csv')

covariates_file = requests.get('https://personal.utdallas.edu/~gcd/data/tract_covariates.csv', allow_redirects=True)
open('tract_covariates.csv', 'wb').write(covariates_file.content)
covariates = pd.read_csv('tract_covariates.csv')

# Build ID from 'state'+'county'+'tract' identifiers
outcomes['id'] = outcomes.apply(lambda row: str(row.state)+str(row.county)+str(row.tract), axis=1)
covariates['id'] = covariates.apply(lambda row: str(row.state)+str(row.county)+str(row.tract), axis=1)

# Outer join outcomes and covariates, drop duplicate columns, sort by ID
cov_out = pd.merge(covariates, outcomes, on='id', how="outer", suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)').sort_values(by=['id'])
# Drop rows with null outcomes
cov_out = cov_out[cov_out['kfr_pooled_pooled_p1'].notna()].sort_values(['id'])

# Move ID column to be first
move_col = cov_out.pop('id')
cov_out.insert(0, 'id', move_col)

# Download file
cov_out.to_csv('../tract_merged.csv', index=False)
