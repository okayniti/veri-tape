# Loan Tape Profiling Report

- loans.csv: 5,030 rows x 14 cols
- payments.csv: 120,000 rows x 10 cols
- duplicated loan_id rows: 60 (30 distinct ids)

## loans.csv columns
| column | dtype | missing | n_unique |
|---|---|---|---|
| loan_id | object | 0 (0.0%) | 5000 |
| origination_date | object | 0 (0.0%) | 2707 |
| loan_type | object | 0 (0.0%) | 12 |
| term_months | int64 | 0 (0.0%) | 4 |
| region | object | 0 (0.0%) | 16 |
| employment_status | object | 309 (6.14%) | 4 |
| origination_channel | object | 0 (0.0%) | 3 |
| credit_score_at_origination | float64 | 183 (3.64%) | 344 |
| borrower_income_at_origination | float64 | 217 (4.31%) | 4812 |
| loan_amount | float64 | 0 (0.0%) | 5029 |
| dti_at_origination | float64 | 265 (5.27%) | 3307 |
| interest_rate | object | 161 (3.2%) | 3733 |
| default_flag | int64 | 0 (0.0%) | 2 |
| status_at_month_24 | object | 0 (0.0%) | 3 |

## Domain-range violations
- **credit_score_at_origination** expected [300, 850]: 16 violations, examples [0.0, 999.0, -5.0, -5.0, 0.0]
- **interest_rate** expected [0, 40]: 19 violations, examples [-3.5, -3.5, 89.9, 89.9, 89.9]
- **dti_at_origination** expected [0, 1.0]: 955 violations, examples [28.33, 45.66, 33.95, 31.49, 21.96]

## Categorical variants
- **region**: {'m': ['M', 'MIDWEST', 'Midwest', 'midwest'], 'n': ['N', 'NORTHEAST', 'Northeast', 'northeast'], 's': ['S', 'SOUTH', 'South', 'south'], 'w': ['W', 'WEST', 'West', 'west']}
- **loan_type**: {'a': ['AUTO', 'auto', 'auto_loan'], 'h': ['HELOC', 'heloc', 'heloc_loan'], 'm': ['MORTGAGE', 'mortgage', 'mortgage_loan'], 'p': ['PERSONAL', 'personal', 'personal_loan']}