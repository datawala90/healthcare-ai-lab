import os
import random
from datetime import datetime, timedelta

import pandas as pd

from src.utils.config import load_config

CFG_PATH = "config/dqi_config.yaml"


def load_dqi_config():
    return load_config(CFG_PATH)


STATUSES = ["PAID", "DENIED", "PENDING"]

# Normal vs "risky" CPT codes
CPT_NORMAL = ["99213", "99214", "99232"]
CPT_SPIKE = ["99285", "J8499"]  # ER / drug-ish codes


def gen_claim_row(base_date: datetime, drift: bool = False) -> dict:
    """Generate one synthetic claim row."""
    days_offset = random.randint(0, 6)
    claim_date = base_date + timedelta(days=days_offset)

    # Base behaviour
    charge = random.gauss(150, 30)
    paid = max(charge - random.gauss(30, 10), 0)
    status = random.choices(STATUSES, weights=[0.7, 0.1, 0.2])[0]
    cpt = random.choice(CPT_NORMAL)

    if drift:
        # Simulate suspicious pattern: higher charges + risky CPT codes
        charge = random.gauss(280, 60)
        paid = max(charge - random.gauss(40, 20), 0)
        status = random.choices(STATUSES, weights=[0.6, 0.25, 0.15])[0]
        cpt = random.choice(CPT_SPIKE)

    row = {
        "claim_id": f"C{random.randint(100000, 999999)}",
        "claim_date": claim_date.strftime("%Y-%m-%d"),
        "member_id": f"M{random.randint(1000, 9999)}",
        "provider_id": f"P{random.randint(1, 40):03d}",
        "cpt_code": cpt,
        "charge_amount": round(charge, 2),
        "paid_amount": round(paid, 2),
        "status": status,
    }

    return row


def main():
    random.seed(42)
    cfg = load_dqi_config()

    baseline_path = cfg.paths.baseline_claims
    current_path = cfg.paths.current_claims

    # Ensure directories exist
    os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
    os.makedirs(os.path.dirname(current_path), exist_ok=True)

    # ---- Baseline week: relatively clean data ----
    baseline_rows = []
    base_start = datetime(2024, 11, 1)
    for _ in range(600):
        baseline_rows.append(gen_claim_row(base_start, drift=False))
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(baseline_path, index=False)

    # ---- Current week: volume drop + drift / nulls ----
    current_rows = []
    curr_start = datetime(2024, 11, 8)
    for _ in range(400):  # fewer rows = volume drop
        row = gen_claim_row(curr_start, drift=True)
        # introduce some null member_ids to simulate DQ issue
        if random.random() < 0.15:
            row["member_id"] = None
        current_rows.append(row)
    current_df = pd.DataFrame(current_rows)
    current_df.to_csv(current_path, index=False)

    print(f"Baseline written to: {baseline_path}  shape={baseline_df.shape}")
    print(f"Current  written to: {current_path}   shape={current_df.shape}")


if __name__ == "__main__":
    main()
