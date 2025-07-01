#!/usr/bin/env python
"""
Run with:  OPENROUTER_API_KEY=sk-or-... python eda_ai.py path/to/file.csv -o report.md
"""

import argparse
import os
import textwrap
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dotenv import load_dotenv                      # pip install python-dotenv
from openai import OpenAI                           # pip install openai

# ─────────────────────────────────────────────────────────────────────────────
# 🔐 Load API key from .env or environment variable
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise EnvironmentError("❌ Set OPENROUTER_API_KEY in your environment or .env file.")

# ─────────────────────────────────────────────────────────────────────────────
# 📡 Setup OpenRouter client (not OpenAI!)
# ─────────────────────────────────────────────────────────────────────────────
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

EXTRA_HEADERS = {
    "HTTP-Referer": "https://github.com/your-org/eda-ai",  # Customize for OpenRouter attribution
    "X-Title": "EDA AI Script",
}

# ─────────────────────────────────────────────────────────────────────────────
# ⚙️ Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODEL = "mistralai/mistral-small-3.2-24b-instruct-2506:free"
MAX_ROWS = 10
N_TOP_COLUMNS = 20
PLOT_DIR = Path("eda_plots")
PLOT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 📥 Load CSV
# ─────────────────────────────────────────────────────────────────────────────
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 📊 Profile basic stats + correlation
# ─────────────────────────────────────────────────────────────────────────────
def basic_profile(df):
    stats = df.describe(include="all").T.round(3)
    nulls = df.isna().sum()
    dtypes = df.dtypes

    profile = pd.concat([stats, nulls.rename("null_cnt"), dtypes.rename("dtype")], axis=1)

    if profile.shape[0] > N_TOP_COLUMNS:
        profile = profile.head(N_TOP_COLUMNS)

    corr = df.select_dtypes("number").corr().round(3)

    return profile, corr

# ─────────────────────────────────────────────────────────────────────────────
# 📈 Generate visual plots
# ─────────────────────────────────────────────────────────────────────────────
def save_plots(df):
    figs = []
    num_cols = df.select_dtypes("number").columns

    if len(num_cols) >= 2:
        plt.figure()
        sns.heatmap(df[num_cols].corr(), annot=False)
        plt.title("Numeric Correlation Heatmap")
        plt.tight_layout()
        fig_path = PLOT_DIR / "heatmap_corr.png"
        plt.savefig(fig_path)
        plt.close()
        figs.append(fig_path)

    for col in num_cols[:3]:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        fig_path = PLOT_DIR / f"hist_{col}.png"
        plt.savefig(fig_path)
        plt.close()
        figs.append(fig_path)

    return figs

# ─────────────────────────────────────────────────────────────────────────────
# 🧠 Craft prompt for LLM
# ─────────────────────────────────────────────────────────────────────────────
# def craft_prompt(df_sample, profile, corr, MAX_ROWS=10, N_TOP_COLUMNS=10):
#     prompt = textwrap.dedent(f"""
#     You are a senior data-analysis consultant engaged by an enterprise client.
#     Below is a snapshot of the dataset we’re investigating, followed by basic
#     profiling artifacts.

#     –––––– DATA SNAPSHOT ––––––
#     1. 🔍 **Sample rows (top {MAX_ROWS}):**
#     ```
#     {df_sample.to_string(index=False)}
#     ```

#     2. 🧾 **Summary statistics (first {N_TOP_COLUMNS} columns):**
#     ```
#     {profile.to_string()}
#     ```

#     3. 🔗 **Numeric correlations:**
#     ```
#     {corr.to_string()}
#     ```
#     ––––––––––––––––––––––––––––

#     ## 📋 REQUIRED REPORT STRUCTURE
#     Generate a **comprehensive, data and insight-heavy EDA** in Markdown with the
#     following numbered sections (use the same headings). Keep your tone
#     concise but authoritative—write as you would for a VP of Data.

#     1. **Executive Summary**
#        * 3–4 key findings in plain business language. Ensure you have proper data points backing it from the analysis. 
#        * Emphasize insights relevant to cost, risk, or revenue.

#     2. **Dataset Overview**
#        * Shape (rows × columns), time range, column groupings (IDs, metrics, dates, categories).
#        * Describe what the dataset represents.

#     3. **Data Quality & Governance**
#        | Check              | Columns Affected | % of Rows and numbers supporting the same | Recommended Action       |
#        |--------------------|------------------|-------------------------------------------|--------------------------|
#        | Missing / Nulls    |                  |                                           |                          |
#        | Duplicates         |                  |                                           |                          |
#        | Type mismatches    |                  |                                           |                          |
#        | High cardinality   |                  |                                           |                          |
#        Mention auditability, PII sensitivity, and regulatory considerations.

#     4. **Univariate Analysis** -- proper data points and formulas backing the information and its derivation as well
#        * Numeric: distributions, skew, outliers (flag z-score > 3 or IQR outliers). 
#        * Categorical: top categories, rare values, one-hot/target encoding strategies.
#        * Temporal: trends, gaps, seasonality if date columns exist.

#     5. **Bivariate & Multivariate Relationships** proper data points and formulas backing the information and its derivation as well
#        * Identify strong and weak correlations.
#        * Use ANOVA or non-parametric tests for categorical↔numeric.
#        * Mention multicollinearity or redundant features (VIF > 10).

#     6. **Target Variable Analysis** *(skip if not known)* proper data points and formulas backing the information and its derivation as well
#        * Class balance (if classification) or skew (if regression).
#        * Identify features most predictive of the target.
#        * Business implications of imbalance or noise.

#     7. **Segment-Level Insights** proper data points and formulas backing the information and its derivation as well
#        * Slice by key dimensions (e.g., geography, business line).
#        * Highlight segments with distinct patterns or anomalies.

#     8. **Feature Engineering Ideas** proper data points and formulas backing the information and its derivation as well
#        * Encoding suggestions, bins, ratios, rolling stats, interactions.
#        * Any external data worth integrating.

#     9. **Risk, Bias & Compliance** proper data points and formulas backing the information and its derivation as well
#        * Detect bias or proxies for sensitive attributes.
#        * Drift warnings (time-based inconsistencies or anomalies).
#        * Note on model explainability or fairness checks.

#     10. **Recommended Next Steps** proper data points and formulas backing the information and its derivation as well
#         * Cleaning/imputation tasks.
#         * Visuals to plot and questions to explore.
#         * Modeling strategy outline (split, CV, metrics).
#         * Dashboard/monitoring setup or delivery plan.

#     ### Formatting Instructions
#     * Format the report in Markdown.
#     * Use bullet points and tables where appropriate.
#     * Provide descriptive summaries, not just raw stats.
#     * Keep total word count ~1,200 words. Do not include plots.

#     Return only the Markdown report. No additional commentary.
#     """)
#     return prompt

import textwrap

def craft_prompt(df_sample, profile, corr, MAX_ROWS=10, N_TOP_COLUMNS=10):
    prompt = textwrap.dedent(f"""
    You are a senior data analysis consultant preparing a business-readable report
    based on an enterprise dataset. This report will be reviewed by executives, product leaders,
    analysts, and compliance teams—assume they are *not* technical.

    Below is the snapshot and profiling information from the dataset. Use this to generate a detailed
    yet understandable EDA report following the structure below.

    –––––– DATA SNAPSHOT ––––––
    1. 📌 **Sample Rows (Top {MAX_ROWS}):**
    ```
    {df_sample.to_string(index=False)}
    ```

    2. 📊 **Summary Statistics (First {N_TOP_COLUMNS} Columns):**
    ```
    {profile.to_string()}
    ```

    3. 📈 **Numeric Correlations:**
    ```
    {corr.to_string()}
    ```

    –––––– END SNAPSHOT ––––––

    ## 📋 REQUIRED EDA STRUCTURE

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


# ─────────────────────────────────────────────────────────────────────────────
# 🤖 Query OpenRouter (Mistral) via OpenAI client
# ─────────────────────────────────────────────────────────────────────────────
def query_llm(prompt: str) -> str:
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=1200,
        extra_headers=EXTRA_HEADERS
    )
    return completion.choices[0].message.content

# ─────────────────────────────────────────────────────────────────────────────
# 📝 Write markdown report
# ─────────────────────────────────────────────────────────────────────────────
def write_report(md_path, narrative, images):
    md = Path(md_path)
    with md.open("w", encoding="utf-8") as f:
        f.write("# 🧾 Automated EDA Report\n\n")
        f.write(narrative)
        if images:
            f.write("\n\n---\n\n## 📉 Plots\n")
            for img in images:
                f.write(f"![plot]({img})\n\n")
    print(f"✅ Report written to {md.resolve()}")

# ─────────────────────────────────────────────────────────────────────────────
# 🚀 Main entry
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AI-assisted Exploratory Data Analysis")
    parser.add_argument("csv", help="Path to CSV file")
    parser.add_argument("-o", "--out", default="eda_report.md", help="Markdown output file")
    args = parser.parse_args()

    df = load_data(args.csv)
    images = save_plots(df)
    profile, corr = basic_profile(df)
    sample = df.head(MAX_ROWS)

    prompt = craft_prompt(sample, profile, corr)
    narrative = query_llm(prompt)

    write_report(args.out, narrative, images)

if __name__ == "__main__":
    main()