from io import BytesIO, TextIOWrapper, StringIO
from zipfile import ZipFile
import re
import pandas as pd
import requests
from pathlib import Path

URL_SIZE_DECILES = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Portfolios_Formed_on_ME_CSV.zip"
URL_FF3 = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

def _download_zip(url: str) -> ZipFile:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return ZipFile(BytesIO(r.content))

def _read_csv_text(z: ZipFile) -> str:
    names = [n for n in z.namelist() if n.lower().endswith(".csv")]
    if not names:
        raise RuntimeError("No CSV found inside the zip.")
    with z.open(names[0]) as f:
        # Fama–French CSVs use latin1 and include long headers/footers
        return TextIOWrapper(f, encoding="latin1").read()

def _extract_monthly_table(csv_text: str) -> pd.DataFrame:
    """
    Find the first data row that looks like YYYYMM,<...>
    Use the immediately preceding comma-separated line as the header,
    then collect contiguous YYYYMM rows until a line breaks the pattern.
    """
    import re
    from io import StringIO

    lines = csv_text.splitlines()

    # index of first YYYYMM row
    first_data_idx = None
    yyyy_mm = re.compile(r"^\s*\d{6}\s*,")
    for i, line in enumerate(lines):
        if yyyy_mm.match(line):
            first_data_idx = i
            break
    if first_data_idx is None:
        raise RuntimeError("Couldn't find any YYYYMM data row in the CSV text.")

    # find header line just above it (must be comma-separated)
    header_idx = None
    for j in range(first_data_idx - 1, -1, -1):
        if "," in lines[j]:
            header_idx = j
            break
    if header_idx is None:
        raise RuntimeError("Couldn't locate a header line before the data block.")

    # collect contiguous YYYYMM rows starting at first_data_idx
    data_lines = [lines[header_idx]]  # header first
    for line in lines[first_data_idx:]:
        if yyyy_mm.match(line):
            data_lines.append(line)
        else:
            break

    # load with pandas
    tbl_text = "\n".join(data_lines)
    return pd.read_csv(StringIO(tbl_text))


def _to_month_period(df: pd.DataFrame) -> pd.DataFrame:
    date_col = df.columns[0]
    df[date_col] = pd.PeriodIndex(df[date_col].astype(str), freq="M")
    df = df.set_index(date_col)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    return df

def download_size_deciles() -> pd.DataFrame:
    z = _download_zip(URL_SIZE_DECILES)
    csv_text = _read_csv_text(z)
    raw_tbl = _extract_monthly_table(csv_text)
    dfm = _to_month_period(raw_tbl)

    # Keep 10 size portfolios (ME1..ME10) if present
    if len(dfm.columns) >= 10:
        dfm = dfm.iloc[:, :10]
        dfm.columns = [f"ME{i}" for i in range(1, 11)]

    # Save
    (DATA_RAW / "Portfolios_Formed_on_ME_raw.csv").write_text(csv_text, encoding="latin1")
    dfm.to_csv(DATA_PROCESSED / "ff_size_deciles_monthly.csv")
    return dfm

def download_ff3() -> pd.DataFrame:
    z = _download_zip(URL_FF3)
    csv_text = _read_csv_text(z)
    raw_tbl = _extract_monthly_table(csv_text)
    dfm = _to_month_period(raw_tbl)

    keep = [c for c in ["Mkt-RF", "SMB", "HML", "RF"] if c in dfm.columns]
    dfm = dfm[keep]

    (DATA_RAW / "F-F_Research_Data_Factors_raw.csv").write_text(csv_text, encoding="latin1")
    dfm.to_csv(DATA_PROCESSED / "ff3_monthly.csv")
    return dfm

def main():
    print("Downloading Fama–French size deciles (monthly)…")
    me = download_size_deciles()
    print(f"  -> saved data/processed/ff_size_deciles_monthly.csv  ({me.shape[0]} months, {me.shape[1]} cols)")

    print("Downloading Fama–French 3 Factors (monthly)…")
    ff3 = download_ff3()
    print(f"  -> saved data/processed/ff3_monthly.csv  ({ff3.shape[0]} months, {ff3.shape[1]} cols)")

    joined = me.join(ff3, how="inner")
    joined.to_csv(DATA_PROCESSED / "ff_size_deciles_with_ff3_monthly.csv")
    print(f"  -> merged file data/processed/ff_size_deciles_with_ff3_monthly.csv  ({joined.shape[0]} rows)")

if __name__ == "__main__":
    main()