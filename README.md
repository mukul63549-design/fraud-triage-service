
# Razorpay Merchant Risk Desk — Fraud Triage Service

## Overview

This project builds a fraud triage service using synthetic transaction data. It cleans transaction records, trains a fraud detection model, retrieves relevant fraud policies, generates validated case notes, and exposes the workflow through a FastAPI service.

The service supports transaction scoring, risk review decisions, policy-based case notes, request logging, and basic operational metrics.

## Reproduce

Requirements: Python 3.11 or compatible with the pinned dependencies.

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Generate the synthetic data:

   ```bash
   python generate_data.py
   ```

4. Run Part A:

   ```bash
   python prepare.py
   ```

5. Run Part B:

   ```bash
   python train.py
   ```

6. Run Part C policy retrieval and case-note checks:

   ```bash
   python retrieval.py
   python casenote.py
   ```

7. Start the API:

   ```bash
   uvicorn app:app --host 127.0.0.1 --port 8000
   ```

8. In a separate terminal, run the smoke tests:

   ```bash
   python smoke_test.py
   ```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

The generated data is synthetic. Run the data-generation script before running the pipeline.

## Part A results

### Raw-data audit

The generator reported 6,090 transaction rows. The raw export contained duplicate records and inconsistent or missing values.

| Check | Result |
|---|---:|
| Raw rows | 6,090 |
| Raw columns | 14 |
| Duplicate rows found | 90 |
| Rows after deduplication | 6,000 |
| Missing transaction amounts | 76 |
| Missing distance values | 193 |
| Out-of-range values found | 5 |
| Raw category spellings | 24 |
| Raw city spellings | 24 |

### Cleaned data

After cleaning and feature engineering, 5,919 rows remained.

| Check | Result |
|---|---:|
| Cleaned rows | 5,919 |
| Remaining duplicate rows | 0 |
| Missing distance values | 191 |
| Merchant categories | 6 |
| Cities | 6 |

The cleaned merchant categories were Electronics, Fashion, Food Delivery, Grocery, Travel, and Utilities. Payment modes were UPI, Card, NetBanking, and Wallet.

### Engineered features

The pipeline adds `is_night`, `high_value`, and `amount_vs_avg_ratio`. The maximum observed `amount_vs_avg_ratio` was 13.11.

The cleaned dataset contained 366 fraud cases, approximately 6.18% of the rows. Predicting every transaction as non-fraud would achieve approximately 93.82% accuracy but detect no fraud cases, so accuracy alone is not an appropriate measure of fraud-model usefulness.

The `chargeback_filed` field is an outcome-related field and must not be used as a real-time prediction input.

## Part B results

### Data split

| Split | Rows | Fraud rows |
|---|---:|---:|
| Training | 3,551 | 220 |
| Validation | 1,184 | 73 |
| Test | 1,184 | 73 |

The training-set median for `distance_from_home_km` was 12.6. Preprocessing and feature encoding were fitted using training data, and the resulting model input contained 26 encoded columns.

### Model comparison and tuning

| Experiment | Precision | Recall | F1 |
|---|---:|---:|---:|
| Logistic regression, threshold 0.5 | 0.625 | 0.342 | 0.442 |
| Balanced logistic regression | 0.252 | 0.781 | 0.381 |
| Oversampled training data | 0.257 | 0.781 | 0.386 |

The balanced logistic regression and oversampling experiments improved fraud recall compared with the plain logistic-regression baseline.

Five-fold cross-validation produced a mean F1 score of 0.402 (standard deviation 0.015) for balanced logistic regression and 0.453 (standard deviation 0.052) for the untuned random-forest experiment.

The tuned random forest used 12 parameter combinations across five folds. Its best parameters were `max_depth=None` and `min_samples_leaf=5`, with a mean cross-validation F1 score of 0.465.

### Threshold selection and test results

The validation threshold was selected as 0.4 to satisfy the target recall constraint of at least 0.65 while choosing the highest qualifying threshold.

| Test metric | Result |
|---|---:|
| Threshold | 0.4 |
| Precision | 0.281 |
| Recall | 0.781 |
| F1 | 0.413 |
| True negatives | 965 |
| False positives | 146 |
| False negatives | 16 |
| True positives | 57 |

The model detects many fraudulent transactions, but its precision is relatively low. A substantial number of legitimate transactions are also flagged for review.

### Data leakage experiment

Including `chargeback_filed` produced a validation F1 score of 0.787 at a threshold of 0.5.

This is a leakage demonstration, not a valid production result. `chargeback_filed` is post-outcome information and must not be used as a real-time model input.

### Top 5 feature importance

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `amount_vs_avg_ratio` | 0.2700 |
| 2 | `txn_count_last_24h` | 0.1779 |
| 3 | `amount_inr` | 0.1339 |
| 4 | `account_age_days` | 0.0835 |
| 5 | `distance_from_home_km` | 0.0709 |

## Part C results

### Policy retrieval

The policy retrieval component uses TF-IDF and cosine similarity to find the most relevant fraud policy.

- Supplied retrieval queries tested: 10
- Retrieval tests passed: 10/10
- Relevant queries returned the expected policy IDs.
- Unrelated cafeteria-menu and email-password queries returned no policy (`None`).

### Case-note generation and validation

The case-note component generates structured JSON notes using the supplied offline `mock_llm.py`.

- Supplied case scenarios processed: 10
- Required fields checked: `summary`, `priority`, `recommended_team`, `recommended_action`, and `policy_id`.
- Invalid priority values are rejected by validation.
- Policy P4 uses the required `block_and_notify` action.
- Invalid responses trigger retries, followed by a conservative fallback when retries are exhausted.
- A simulated language-model connection error is handled with a safe fallback.
- Python syntax check: `python -m py_compile casenote.py` completed without errors.

## Part D results

### API endpoints

The FastAPI application exposes the following endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Reports service health and model-loading status |
| `POST /score` | Scores a transaction and returns its fraud probability, threshold, and decision |
| `POST /case` | Scores a transaction, retrieves a relevant policy, and generates a validated case note |
| `GET /metrics` | Returns basic request metrics and the active fraud threshold |

The API uses Pydantic request models to validate transaction inputs. Model features are aligned to the training feature columns, and missing distance values are handled using the training-set median.

The scoring response includes a fraud probability and the selected threshold of 0.4. A high-risk override can flag a transaction when the amount is at least INR 50,000 and the device is new.

### Smoke-test results

The smoke-test script completed all 11 checks successfully.

| Check | Result |
|---|---|
| Health endpoint | PASS |
| Metrics endpoint | PASS |
| Score endpoint | PASS |
| Fraud probability range | PASS |
| Threshold range | PASS |
| Decision field | PASS |
| High-risk override field | PASS |
| Case endpoint | PASS |
| Required case-note fields | PASS |
| Invalid amount rejected | PASS |
| Missing required fields rejected | PASS |

**Final smoke-test result: 11/11 checks passed.**

### Logging and metrics

The service writes request logs to `service.log`. Manual POST requests to the scoring and case endpoints returned successful responses and were recorded in the log.

The metrics endpoint was tested and returned request counts, error counts, and the active threshold. The observed manual-test response reported zero errors.

## Limitations

- The dataset is synthetic and does not establish real-world fraud detection performance.
- Fraud cases are a minority of the cleaned dataset, so precision and recall must be considered alongside accuracy.
- At the selected threshold, the model produces false positives that require human review.
- The high-risk override is a fixed rule and should be evaluated against real operational requirements before production use.
- Policy retrieval uses TF-IDF and cosine similarity; relevance may be limited for queries that differ substantially from the policy wording.
- Case notes use the supplied mock language model, not a production language-model service.
- The API and its request metrics are intended for a capstone demonstration, not a production deployment.
- The leakage experiment is for demonstration only. `chargeback_filed` must not be used as a real-time model feature.