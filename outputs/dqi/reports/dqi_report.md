# Agentic Data Quality Investigator Report

Project: **agentic_dqi_healthcare**

## Volume

- Baseline rows: `600`
- Current rows: `400`
- Volume change: `-33.33%`

## Numeric Mean Drift

- **charge_amount** mean change: `91.64%`
- **paid_amount** mean change: `108.10%`

## Null Drift (increase in null ratio)

- **member_id** nulls increased by `16.25%`

## Categorical Distribution Changes

- **status**:
  - `DENIED` ↑ `17.25%`
  - `PAID` ↓ `-13.17%`
- **cpt_code**:
  - `99232` ↓ `-34.67%`
  - `99285` ↑ `50.25%`
  - `J8499` ↑ `49.75%`
  - `99213` ↓ `-31.67%`
  - `99214` ↓ `-33.67%`

---

## LLM Analysis

**Severity:** `MEDIUM`

**Summary:** {'data_drift': True, 'type': ['volume', 'nulls', 'amounts']}


**Suspected Causes:**
- ETL issue
- upstream system change
- real business event

**Recommended Checks:**
- Verify ETL pipeline for data quality and integrity
- Check upstream systems for changes or updates that may be affecting data
- Monitor business events and adjust data processing accordingly

**Business Risk:**
Potential impact on claims processing and payment accuracy