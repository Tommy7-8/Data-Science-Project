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

OUT = DATA_PROCESSED / "ff_size_deciles_with_ff3_monthly_decimals.csv"


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


def fmt_num(val, max_decimals=6, dec_char="."):
    """Format numbers for pretty aligned CSV, without unnecessary trailing zeros."""
    if pd.isna(val):
        return ""
    # start with fixed decimal representation
    s = f"{val:.{max_decimals}f}"
    # remove trailing zeros and possible trailing decimal point
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    # replace decimal char if needed
    if dec_char != ".":
        s = s.replace(".", dec_char)
    return s



def download_size_deciles() -> pd.DataFrame:
    """
    Download the 'Portfolios Formed on ME' table and extract the 10 *decile* portfolios only:
    Lo 10, 2-Dec, 3-Dec, ..., 9-Dec, Hi 10  -> rename to ME1..ME10.
    """
    import re
    z = _download_zip(URL_SIZE_DECILES)
    csv_text = _read_csv_text(z)
    raw_tbl = _extract_monthly_table(csv_text)

    # save full raw text to data/raw
    (DATA_RAW / "Portfolios_Formed_on_ME_raw.csv").write_text(csv_text, encoding="latin1")

    # Normalize column names for robust matching
    cols = [str(c).strip() for c in raw_tbl.columns]
    raw_tbl.columns = cols

    # Find decile block columns (works with "Lo 10", "2-Dec", ..., "9-Dec", "Hi 10")
    wanted = []
    decile_labels = ["Lo 10"] + [f"{i}-Dec" for i in range(2, 10)] + ["Hi 10"]
    for lab in decile_labels:
        # find first column that contains this label as substring (be tolerant to spaces)
        match = next(
            (c for c in cols if re.search(rf"\b{re.escape(lab)}\b", c, flags=re.IGNORECASE)),
            None,
        )
        if match is None:
            raise RuntimeError(f"Couldn't find decile column '{lab}' in headers: {cols[:15]} ...")
        wanted.append(match)

    # Keep only the date + these 10 decile columns
    date_col = raw_tbl.columns[0]
    keep_cols = [date_col] + wanted
    df = raw_tbl[keep_cols].copy()

    # Convert to PeriodIndex monthly, numeric columns
    df[date_col] = pd.PeriodIndex(df[date_col].astype(str), freq="M")
    df = df.set_index(date_col)
    for c in wanted:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Rename to ME1..ME10 (ME1 = Lo 10, ME10 = Hi 10)
    rename_map = {wanted[0]: "ME1"}
    for i, c in enumerate(wanted[1:-1], start=2):  # 2-Dec .. 9-Dec -> ME2..ME9
        rename_map[c] = f"ME{i}"
    rename_map[wanted[-1]] = "ME10"
    df = df.rename(columns=rename_map)

    # Replace French missing flag -99.99 with NaN, then drop rows where all ME* are NaN
    for c in df.columns:
        df[c] = df[c].replace(-99.99, pd.NA)
    df = df.dropna(how="all")

    return df


def download_ff3() -> pd.DataFrame:
    z = _download_zip(URL_FF3)
    csv_text = _read_csv_text(z)
    raw_tbl = _extract_monthly_table(csv_text)
    dfm = _to_month_period(raw_tbl)

    # save full raw text to data/raw
    (DATA_RAW / "F-F_Research_Data_Factors_raw.csv").write_text(csv_text, encoding="latin1")

    keep = [c for c in ["Mkt-RF", "SMB", "HML", "RF"] if c in dfm.columns]
    dfm = dfm[keep]
    return dfm


def main():
    print("Downloading Fama–French size deciles (monthly)…")
    me = download_size_deciles()
    print(f"  -> fetched size deciles  ({me.shape[0]} months, {me.shape[1]} cols); raw saved in data/raw")

    print("Downloading Fama–French 3 Factors (monthly)…")
    ff3 = download_ff3()
    print(f"  -> fetched FF3 factors  ({ff3.shape[0]} months, {ff3.shape[1]} cols); raw saved in data/raw")

    # Merge on the monthly index
    joined = me.join(ff3, how="inner")
    print(f"  -> merged table in memory  ({joined.shape[0]} rows, {joined.shape[1]} cols, still in PCT)")

    # ---- replicate prepare_data.py: convert to decimals + pretty aligned CSV ----
    df = joined.copy()

    # parse index as monthly period (robust, even if already PeriodIndex)
    try:
        df.index = pd.PeriodIndex(df.index.astype(str), freq="M")
    except Exception:
        df.index = pd.to_datetime(df.index.astype(str), format="%Y-%m", errors="coerce")

    # convert to numeric and go from percent to decimal
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0
    df.index.name = "date"

    # drop any stray 'Unnamed:*' cols
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Build aligned strings table for pretty CSV (semicolon-separated)
    df_txt = df.reset_index().copy()
    for c in df_txt.columns:
        if c != "date":
            df_txt[c] = df_txt[c].map(lambda v: fmt_num(v))

    # compute column widths (max of header and values)
    widths = {c: max(len(c), df_txt[c].astype(str).map(len).max()) for c in df_txt.columns}

    # header (date left-aligned, numbers right-aligned)
    header_cells = [f"{'date':<{widths['date']}}"] + [
        f"{c:>{widths[c]}}" for c in df_txt.columns if c != "date"
    ]

    # rows
    row_lines = []
    for _, row in df_txt.iterrows():
        left = f"{str(row['date']):<{widths['date']}}"
        nums = [f"{str(row[c]):>{widths[c]}}" for c in df_txt.columns if c != "date"]
        row_lines.append([left] + nums)

    # Save visually aligned CSV (semicolon separator)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines_csv = [";".join(header_cells)] + [";".join(cells) for cells in row_lines]
    OUT.write_text("\n".join(lines_csv), encoding="utf-8-sig")
    print("Saved visually aligned decimal CSV →", OUT)


if __name__ == "__main__":
    main()
