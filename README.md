# EDA-AI: Automated Exploratory Data Analysis with LLMs

---

## Overview

**EDA-AI** is a command-line tool that automatically generates a clean, human-readable Exploratory Data Analysis (EDA) report for any CSV dataset — powered by large language models like **Mistral** via **OpenRouter**.

It’s designed for **data professionals**, **business analysts**, and **product teams** who need quick, insightful overviews of their data without manually coding EDA logic or writing reports.

---

## Problem Statement

> **How can we automate exploratory data analysis and make the results understandable to non-technical stakeholders using AI?**

Performing EDA is time-consuming and often too technical for business users to interpret.

**EDA-AI** solves this by:
- Extracting core statistics and visual patterns from the dataset
- Structuring a prompt to guide the LLM (e.g., Mistral) to generate a business-friendly Markdown report
- Producing charts and summaries you can drop into presentations or dashboards

---

##  What It Does

-  **Load Data**: Accepts any `.csv` file as input  
-  **Profile Data**: Computes descriptive stats, correlation, and sample views  
-  **Generate Visuals**: Saves key plots like histograms and heatmaps  
-  **LLM-Powered Analysis**: Crafts a detailed prompt and sends it to an AI model via OpenRouter  
-  **Write Markdown Report**: Outputs a clean, structured EDA report with insights and optional plots

---

##  Example Use Case

```bash
OPENROUTER_API_KEY=sk-or-... python eda_ai.py data/customer_data.csv -o report.md
```

## LLM Prompt Template
```
def craft_prompt(df_sample, profile, corr, MAX_ROWS=10, N_TOP_COLUMNS=10):
    prompt = textwrap.dedent(f"""
    You are a senior data analysis consultant preparing a business-readable report
    based on an enterprise dataset. This report will be reviewed by executives, product leaders,
    analysts, and compliance teams—assume they are *not* technical.

    Below is the snapshot and profiling information from the dataset. Use this to generate a detailed
    yet understandable EDA report following the structure below.

    –––––– DATA SNAPSHOT ––––––
    1.  **Sample Rows (Top {MAX_ROWS}):**
    ```
    {df_sample.to_string(index=False)}
    ```

    2.  **Summary Statistics (First {N_TOP_COLUMNS} Columns):**
    ```
    {profile.to_string()}
    ```

    3.  **Numeric Correlations:**
    ```
    {corr.to_string()}
    ```

    –––––– END SNAPSHOT ––––––

    ##  REQUIRED EDA STRUCTURE

    Format your response in **Markdown**. Keep it clear, structured, and use everyday language when needed.
    All stats must use **actual numbers**, not just percentages.

    ### 1. **Executive Summary**
    - Summarize 3–5 **high-impact insights** in simple, business-friendly language.
    - Quantify where possible (e.g., "65% of missing revenue values concentrated in Q4").
    - Tie findings to business metrics like cost, risk, growth, or churn.
    - Include 1–2 quick wins or critical watchouts.

    ### 2. **Dataset Overview**
    - Dataset shape (rows × columns)
    - Time coverage (start and end date, if applicable)
    - Data source origin (CRM, internal logs, 3rd-party, etc.)
    - Column grouping by type:
      - **Identifiers**: (e.g., order_id, user_id)
      - **Numeric metrics**: (e.g., price, discount)
      - **Categorical features**: (e.g., product_type)
      - **Dates or temporal fields**

    ### 3. **Data Quality & Governance**
    - Report **exact number of rows** affected per issue.
    - Provide actual **column names** and not placeholders.
    
    | Issue               | Columns Affected                     | Rows Affected (# / %) | Recommended Action             |
    |---------------------|--------------------------------------|------------------------|--------------------------------|
    | Missing Values       | e.g., `age`, `total_spent`           | 1,234 / 12.3%          | Impute median; consider flag   |
    | Duplicates           | e.g., `order_id`                     | 78 / 0.7%              | Drop duplicates                |
    | Type Mismatches      | e.g., `signup_date` as string        | 90 / 0.8%              | Convert to datetime            |
    | High Cardinality     | e.g., `sku_code` (8,123 unique vals) | -                      | Consider frequency grouping    |

    - State if PII/sensitive data is present (e.g., emails, ZIP codes)
    - Mention auditability: Can the data be traced to its source?
    - Regulatory relevance (GDPR, HIPAA, FERPA etc.)

    ### 4. **Univariate Analysis**
    - For **Numeric**:
      - Min, max, mean, median, std dev, skew
      - Outlier flags: Z-score > 3 or IQR method
    - For **Categorical**:
      - Most frequent categories
      - Rare values (frequency < 1%)
    - For **Dates**:
      - Gaps in time, trends, or seasonality patterns

    ### 5. **Bivariate & Multivariate Analysis**
    - Report feature relationships:
      - Numeric ↔ Numeric (correlations)
      - Categorical ↔ Numeric (boxplot-style insights or ANOVA)
      - Categorical ↔ Categorical (contingency tables)
    - Highlight multicollinearity using VIF > 10
    - Detect strong predictors or surprising null relationships

    ### 6. **Target Variable Analysis** *(if applicable)*
    - Clearly define the target column and its role
    - Report balance (classification) or skew (regression)
    - Top predictors using correlation, ANOVA, or decision trees
    - Business implications of skew or noise in target

    ### 7. **Segment-Level Insights**
    - Break analysis by **segments**:
      - Geography, customer tier, time period, product line, etc.
    - Identify standout patterns or problem areas (e.g., "Customer churn highest in Tier C, 32%")

    ### 8. **Feature Engineering Opportunities**
    - List feature creation ideas:
      - Ratios, bins, aggregates, rolling metrics, flags
      - Encoding suggestions for categoricals
      - Date-based features (e.g., time since signup)
    - External datasets worth integrating (e.g., macroeconomic, weather)

    ### 9. **Risk, Bias & Compliance Checks**
    - Detect bias indicators: gender, geography, age
    - Any proxy variables that may leak sensitive info
    - Drift or time-based anomalies in key variables
    - Model fairness and explainability considerations

    ### 10. **Next Steps & Recommendations**
    - Data prep: Cleaning, outlier handling, imputations
    - Visuals to create (e.g., trend lines, distributions, tree maps)
    - Modeling plan: train-test split, CV strategy, metrics to use
    - Monitoring plans or dashboards to implement
    - Stakeholders to involve (e.g., legal, ops, product)

    ### 11. **Limitations & Caveats**
    - Gaps in data, suspected errors, unvalidated assumptions
    - What cannot be concluded from the data
    - Risks in overinterpreting small segments

    ### 🔧 Report Instructions
    - Use clear **Markdown format**
    - Use **real numbers, column names, and exact values**
    - Use tables and bullet points liberally
    - Word count ~1,200
    - **Do not include plots**, but describe what each would show if included

    Return only the Markdown report. Do not include Python code or commentary.
    """)
    return prompt
```

---

## Example Files

-  [Sample Input File: `voice.csv`](/voice.csv)
-  [Generated EDA Report: `report1.md`](/report1.md)

## This generates:

 -  report.md: A rich Markdown report ready for business review
 -  eda_plots/: Optional folder of charts (e.g., correlation heatmap)

---

##  Output Highlights:
- Executive Summary: Key trends, risks, and quick wins
- Data Quality Table: Exact row counts and affected columns
- Univariate & Bivariate Analysis: Numeric, categorical, and time insights
- Segment-Level Patterns: E.g., churn or spend by region or product
- Feature Engineering Suggestions
- Risk, Bias & Compliance Checks
- Recommended Next Steps for modeling

--- 

##  Tech Stack:
- Python
- Pandas, Seaborn, Matplotlib
- LLM via OpenRouter (mistralai/mistral-small-3.2)
- Markdown output
- Command-line interface

---


##  Requirements:
Python 3.8+
.env file with OPENROUTER_API_KEY or set the key in your environment

---

## Python packages:
- openai
- pandas
- matplotlib
- seaborn
- tqdm
- python-dotenv

##  Ideal For:
- Analysts looking to speed up EDA
- Data science teams preparing datasets for modeling
- Product managers or executives who want a quick, readable data summary

