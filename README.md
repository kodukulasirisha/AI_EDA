📊 EDA-AI: Automated Exploratory Data Analysis with LLMs
🧠 Overview
EDA-AI is a command-line tool that automatically generates a clean, human-readable Exploratory Data Analysis (EDA) report for any CSV dataset — powered by large language models like Mistral via OpenRouter.

It’s designed for data professionals, business analysts, and product teams who need quick, insightful overviews of their data without manually coding EDA logic or writing reports.

🎯 Problem Statement
How can we automate exploratory data analysis and make the results understandable to non-technical stakeholders using AI?

Performing EDA is time-consuming and often too technical for business users to interpret. This project solves that by:
1.Extracting core statistics and visual patterns from the dataset
2.Structuring a prompt to guide the LLM (e.g., Mistral) to generate a business-friendly Markdown report
3.Producing charts and summaries you can drop into presentations or dashboards

🚀 What It Does
📂 Load Data: Accepts any .csv file as input
📊 Profile Data: Computes descriptive stats, correlation, and sample views
📈 Generate Visuals: Saves key plots like histograms and heatmaps
🧠 LLM-Powered Analysis: Crafts a detailed prompt and sends it to an AI model via OpenRouter
📝 Write Markdown Report: Outputs a clean, structured EDA report with insights and optional plots


🧪 Example Use Case
bash
Copy code
OPENROUTER_API_KEY=sk-or-... python eda_ai.py data/customer_data.csv -o report.md
This generates:
report.md: A rich Markdown report ready for business review
/eda_plots/: Optional folder of charts (e.g., correlation heatmap)


📦 Output Highlights
🔹 Executive Summary: Key trends, risks, and quick wins

🔹 Data Quality Table: Exact row counts and affected columns

🔹 Univariate & Bivariate Analysis: Numeric, categorical, and time insights

🔹 Segment-Level Patterns: E.g., churn or spend by region or product

🔹 Feature Engineering Suggestions

🔹 Risk, Bias & Compliance Checks

🔹 Recommended Next Steps for modeling

🛠 Tech Stack
1.Python, Pandas, Seaborn, Matplotlib
2.LLM via OpenRouter (e.g., mistralai/mistral-small-3.2)
3.Markdown output
4.Command-line interface

🔐 Requirements
1.Python 3.8+
2..env file or OPENROUTER_API_KEY set in environment
3.Dependencies: openai, pandas, matplotlib, seaborn, dotenv, tqdm

💡 Ideal For
1.Analysts looking to speed up EDA
2.Data science teams preparing datasets for modeling
3.Product managers or executives who want a quick, readable data summary

