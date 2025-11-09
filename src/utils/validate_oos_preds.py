# src/utils/validate_oos_preds.py
import glob
import os
import pandas as pd

def main():
    files = sorted(glob.glob("results/oos_preds/ME*_oos_preds.csv"))
    if not files:
        raise FileNotFoundError("No files found in results/oos_preds/ME*_oos_preds.csv")

    rows = []
    starts, ends, lengths = set(), set(), set()
    problems = []

    for p in files:
        df = pd.read_csv(p)
        name = os.path.basename(p).split("_")[0]  # e.g., 'ME1'
        # basic checks
        if df.isna().any().any():
            problems.append(f"{name}: contains NaNs")

        first_m = df["month"].iloc[0]
        last_m  = df["month"].iloc[-1]
        n       = len(df)

        starts.add(first_m)
        ends.add(last_m)
        lengths.add(n)

        rows.append({"decile": name, "first_month": first_m, "last_month": last_m, "rows": n})

    summary = pd.DataFrame(rows).sort_values("decile")
    print("\nPer-decile OOS coverage:")
    print(summary.to_string(index=False))

    ok = True
    if len(starts) != 1:
        ok = False
        print("\n[WARN] Not all deciles share the same OOS START month:", starts)
    if len(lengths) != 1:
        ok = False
        print("\n[WARN] Not all deciles have the same number of OOS rows:", lengths)
    if len(ends) != 1:
        ok = False
        print("\n[WARN] Not all deciles share the same OOS END month:", ends)

    if problems:
        ok = False
        print("\n[WARN] Problems detected:")
        for msg in problems:
            print(" -", msg)

    # Quick expectation: last month should be 2025-08
    if "2025-08" not in ends:
        ok = False
        print("\n[WARN] Last OOS month is not 2025-08:", ends)

    print("\nOverall status:", "OK ✅" if ok else "Check warnings ⚠️")

if __name__ == "__main__":
    main()