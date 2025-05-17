# 🤖 AI Data Analyst - Workplete Internship Project

This project is a prototype of an **AI employee** designed to assist with **data analysis and reporting**. It can:

- Ingest various data formats (CSV, JSON, Excel)
- Clean and process raw data
- Perform statistical and machine learning analysis
- Generate detailed reports in **Word (.docx)** and optionally **PDF**
- Interact with users via a **simulated command-line interface (CLI)**

---

## 📂 Project Features

### ✅ Data Ingestion
- Accepts file formats: `.csv`, `.json`, `.xls`, `.xlsx`
- Reads data from a user-specified path

### ✅ Data Cleaning
- Standardizes column names
- Handles missing values with forward/backward fill
- Removes duplicates

### ✅ Data Analysis
- Summary statistics and correlation heatmap
- KMeans clustering
- Linear regression with model score and coefficients

### ✅ Report Generation
- Creates a detailed report with:
  - Dataset summary
  - Data types and shape
  - Descriptive statistics
  - Regression results (if any)
  - Correlation heatmap image
- Saves as `output/report.docx`

### ✅ CLI Simulation
- Users can type natural commands like:
  - `summary`
  - `cluster`
  - `regression`
  - `report`
  - `exit`

---

## 🚀 How to Run

### ✅ 1. Install Dependencies

```bash
pip install pandas matplotlib seaborn scikit-learn python-docx openpyxl
