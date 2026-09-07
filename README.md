# Credit Risk Intelligence Platform

A lightweight AI-powered credit risk analytics platform built using the Home Credit Default Risk dataset.

The platform combines:

- Exploratory Data Analysis
- Credit default prediction
- SHAP-based explainability
- Business-readable credit rules
- Natural-language Talk-to-Data
- SQL validation and safe execution
- Streamlit interface
- Dockerized deployment

---

## 1. Problem Statement

Credit institutions need fast and explainable ways to identify applicants who may have a higher risk of loan default.

This project builds a lightweight decision-support platform that:

1. Estimates default probability for an applicant.
2. Converts the probability into Low / Medium / High risk.
3. Explains the prediction using SHAP.
4. Provides business-readable rules derived from observed data patterns.
5. Allows analysts to ask questions about the dataset in natural language.

This is a decision-support system and is not intended to replace human credit decisions.

---

## 2. Architecture

```text
                    ┌──────────────────────┐
                    │   Streamlit UI       │
                    │ Overview / Risk /    │
                    │ Rules / Ask Data     │
                    └──────────┬───────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
       ┌───────────┐     ┌────────────┐    ┌─────────────┐
       │ ML Model  │     │ SHAP       │    │ Rule Engine │
       │ LightGBM  │     │ Explainability│ │             │
       └─────┬─────┘     └──────┬─────┘    └─────────────┘
             │                  │
             └──────────┬───────┘
                        │
                        ▼
              ┌──────────────────┐
              │ Processed         │
              │ Applicant Data    │
              └──────────────────┘

                    Talk-to-Data
                         │
                         ▼
                ┌─────────────────┐
                │ Rule-based SQL  │
                │ + Groq LLM      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ SQL Validator   │
                └────────┬────────┘
                         │
                         ▼
                    ┌─────────┐
                    │ DuckDB  │
                    └────┬────┘
                         │
                         ▼
                Business Answer
````

---

## 3. Dataset

Dataset: Home Credit Default Risk

Main files used during data preparation:

* application_train.csv
* bureau.csv
* previous_application.csv
* installments_payments.csv

The final application-level dataset combines applicant information with aggregated credit-history and repayment features.

The raw dataset is intentionally not included in the Git repository.

---

## 4. Data Preparation

The preprocessing pipeline included:

* Missing-value handling
* Removal of highly sparse/redundant features
* Categorical feature encoding
* Numerical feature imputation
* Feature scaling where required
* Aggregation of historical bureau information
* Aggregation of previous application information
* Aggregation of installment repayment behaviour

Additional engineered features include:

* AGE_YEARS
* EMPLOYMENT_YEARS
* CREDIT_INCOME_RATIO
* ANNUITY_INCOME_RATIO
* CREDIT_GOODS_RATIO
* BUREAU_DEBT_TO_CREDIT
* PREV_REFUSAL_RATE
* PAYMENT_TO_INSTALLMENT_RATIO

Historical features were aggregated at applicant level so that the final prediction model operates on one row per applicant.

---

## 5. Exploratory Data Analysis

The dataset contains 307,511 applications.

Target distribution:

* No Default: 282,686 applications
* Default: 24,825 applications
* Default rate: approximately 8.07%

Important observations from the analysis include:

* The target is strongly imbalanced, with defaults representing a minority of applications.
* Male applicants show a higher observed default rate than female applicants.
* Education level is associated with different observed default rates.
* Previous application refusal behaviour provides useful risk information.
* Payment delay and late-payment behaviour provide useful signals.
* External credit indicators are among the strongest predictive features.

The Streamlit Overview section provides visual summaries of these patterns.

---

## 6. Machine Learning

### Model

LightGBM was selected as the final classification model because it works well with large tabular datasets and mixed feature types.

### Class Imbalance

The target is highly imbalanced.

The training process uses class weighting through:

```text
scale_pos_weight = 11.39
```

This gives greater importance to the minority default class.

### Train / Validation Split

A stratified train-validation split was used to preserve the target distribution.

* Training samples: 246,008
* Validation samples: 61,503

### Evaluation

Validation results at threshold 0.5:

| Metric    | Result |
| --------- | -----: |
| ROC-AUC   | 0.7738 |
| PR-AUC    | 0.2688 |
| Precision | 0.1847 |
| Recall    | 0.6661 |
| F1 Score  | 0.2892 |

An additional threshold analysis identified an F1-oriented threshold of approximately:

```text
0.6622
```

with:

* Precision: 0.2652
* Recall: 0.4187
* F1: 0.3248

The threshold is treated separately from model discrimination performance.

---

## 7. Risk Scoring

For each applicant, the system returns:

* Default probability
* Risk score
* Risk band

Risk bands are presented as:

```text
Low
Medium
High
```

The model probability is converted into a 0–100 risk score for easier business interpretation.

---

## 8. Explainable AI

SHAP TreeExplainer is used to explain individual LightGBM predictions.

For every prediction, the application can display:

* Factors increasing risk
* Factors reducing risk
* SHAP contribution values

Important global predictive features include:

* EXT_SOURCE_3
* EXT_SOURCE_1
* EXT_SOURCE_2
* DAYS_ID_PUBLISH
* DAYS_LAST_PHONE_CHANGE
* AMT_ANNUITY
* DAYS_REGISTRATION
* DAYS_BIRTH
* CREDIT_INCOME_RATIO
* AMT_CREDIT
* ANNUITY_INCOME_RATIO
* AMT_GOODS_PRICE

The application converts technical feature names into more understandable labels for business users.

---

## 9. Business Rules

Rules are derived from observed relationships in the processed dataset rather than being invented manually.

Examples of higher-risk patterns observed in the analysis include:

* Lower external credit indicators are associated with higher observed default risk.
* Higher bureau debt-to-credit ratios are associated with higher observed default risk.
* Higher previous application refusal rates are associated with higher observed default risk.
* Higher average payment delays are associated with higher observed default risk.
* Higher counts of late payments are associated with higher observed default risk.

These rules are intended as interpretable decision-support signals and should not be treated as strict underwriting policies.

---

## 10. Talk-to-Data

The Talk-to-Data component allows users to ask questions in plain English.

Example questions:

```text
What is the overall default rate?
```

```text
Are male applicants more likely to default than female applicants?
```

```text
What is the default rate by education?
```

```text
How many applicants defaulted?
```

```text
What is the average credit amount?
```

```text
What is the default rate by income type?
```

```text
What is the average previous application refusal rate?
```

The system contains deterministic SQL patterns for common business questions and uses Groq for questions outside those predefined patterns.

---

## 11. NL-to-SQL Safety

The generated SQL passes through a validation layer before execution.

The validator:

* Allows only SELECT queries
* Allows only the applications table
* Blocks INSERT
* Blocks UPDATE
* Blocks DELETE
* Blocks DROP
* Blocks ALTER
* Blocks CREATE
* Blocks TRUNCATE
* Blocks ATTACH
* Blocks COPY
* Blocks EXPORT
* Blocks INSTALL
* Blocks PRAGMA

This helps reduce unsafe SQL execution and hallucinated table references.

---

## 12. Prompt Engineering and Token Optimization

The NL-to-SQL prompt explicitly provides:

* Allowed table
* Allowed columns
* Target definition
* Default-rate calculation
* Query restrictions
* Minimum group size for comparisons

The answer-generation prompt instructs the LLM to:

* Answer only from the SQL result
* Avoid invented facts
* Avoid causal claims
* Avoid unnecessary explanations
* Use plain business language
* Avoid Markdown formatting
* Keep responses concise

The response token limit is kept intentionally small for the business-answer layer because these responses only require one or two sentences.

Temperature is set to:

```text
0
```

to make the output more deterministic.

---

## 13. Conversation Memory

The Talk-to-Data interface maintains conversation history during the current Streamlit session.

Users can see:

```text
You: What is the overall default rate?
Assistant: The overall default rate in this dataset is 8.07%.

You: What is the average credit amount?
Assistant: The average credit amount in this dataset is approximately 599,026.
```

A Clear Conversation button is provided to reset the session history.

---

## 14. Project Structure

```text
credit_risk_platform/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── original Home Credit datasets
│   │
│   └── processed/
│       └── final_applicant_data.parquet
│
├── models/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── feature_names.pkl
│   ├── threshold.pkl
│   └── business_rules.json
│
├── notebooks/
│   └── credit_risk_analysis.ipynb
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── __init__.py
│   │
│   ├── ml/
│   │   ├── predict.py
│   │   └── __init__.py
│   │
│   ├── explainability/
│   │   ├── shap_explainer.py
│   │   └── __init__.py
│   │
│   ├── rules/
│   │   └── rule_engine.py
│   │
│   └── talk_to_data/
│       ├── nl_to_sql.py
│       ├── sql_validator.py
│       ├── query_runner.py
│       ├── prompt_templates.py
│       ├── answer_generator.py
│       └── talk_to_data.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 15. Local Setup

Clone the repository:

```bash
git clone <repository-url>
cd credit_risk_platform
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

Make sure the processed dataset exists at:

```text
data/processed/final_applicant_data.parquet
```

Run the application:

```bash
streamlit run app/app.py
```

Open:

```text
http://localhost:8501
```

---

## 16. Docker Setup

Build the application:

```bash
docker compose build
```

Start the application:

```bash
docker compose up
```

Open:

```text
http://localhost:8501
```

The processed data is mounted into the container at runtime rather than copied into the Docker image.

---

## 17. Environment Variables

Required:

```env
GROQ_API_KEY=
```

The real `.env` file must not be committed to Git.

Use `.env.example` as the template.

---

## 18. Saved Artifacts

The trained model artifacts are stored in:

```text
models/
```

including:

* trained LightGBM model
* preprocessing pipeline
* feature names
* decision threshold
* business rules

This allows the application to perform inference without retraining the model.

---

## 19. Limitations

* The model is trained on historical Home Credit data and may not generalize to other lending populations.
* Observed relationships should not automatically be interpreted as causal relationships.
* The Talk-to-Data system supports a constrained analytics schema rather than unrestricted database exploration.
* Conversation memory currently applies to the active Streamlit session.
* SHAP explanations are model explanations and should be interpreted as decision-support signals.
* No external production credit bureau data is used.
* The current system is intended as a lightweight prototype rather than a production underwriting system.

---

## 20. Future Improvements

Possible improvements include:

* Probability calibration
* More extensive hyperparameter optimization
* Fairness and subgroup performance analysis
* Authentication and role-based access
* Persistent conversation history
* More advanced natural-language query coverage
* Automated monitoring for model drift
* Model versioning and experiment tracking
* Production database integration

---

## 21. Disclaimer

This project is a technical demonstration of explainable credit-risk analytics.

It should be used as a decision-support prototype and not as an autonomous system for making real-world lending decisions.

```

### After pasting

Save `README.md`.

Do **not** spend time polishing the README design yet. The content is what matters first.

Then our next step will be the **architecture diagram + presentation PDF**, which are the remaining submission deliverables. :contentReference[oaicite:1]{index=1}
```
