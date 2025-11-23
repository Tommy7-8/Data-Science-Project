"""
Download and preprocess Fama–French size-decile portfolios and FF3 factors.

This script:
- Downloads the original CSV ZIPs from Kenneth R. French's online library
- Extracts the *monthly* data blocks from the messy CSVs
- Cleans and reshapes the size-decile portfolios into ME1..ME10
- Downloads and cleans FF3 factors
- Merges everything and writes a visually aligned CSV (semicolon-separated)
  with returns in *decimals* (not percent).

The public API for the rest of the project is:

- download_size_deciles()
- download_ff3()
- main()

All other helpers are internal.
"""

from io import BytesIO, TextIOWrapper
from zipfile import ZipFile
from pathlib import Path
import re

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants and paths
# ---------------------------------------------------------------------------

# Source URLs from the Kenneth R. French Data Library
URL_SIZE_DECILES = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "Portfolios_Formed_on_ME_CSV.zip"
)
URL_FF3 = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_CSV.zip"
)

# Local data folders
DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# Final processed file (returns in decimals, nicely aligned)
OUT = DATA_PROCESSED / "ff_size_deciles_with_ff3_monthly_decimals.csv"


# ---------------------------------------------------------------------------
# Low-level helpers: download + CSV text extraction
# ---------------------------------------------------------------------------

def _download_zip(url: str) -> ZipFile:
    """
    Download a ZIP file from the given URL and return it as a ZipFile object.

    Raises:
        requests.HTTPError: if the HTTP request fails.
    """
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return ZipFile(BytesIO(r.content))


def _read_csv_text(z: ZipFile) -> str:
    """
    Read the *first* CSV file inside a ZipFile and return its contents as text.

    Notes:
        - Fama–French CSVs use 'latin1' encoding.
        - They contain long headers and footers; we keep the whole text and
          later extract only the monthly data block we care about.
    """
    # Get all CSV files inside the ZIP; we assume the first one is the table we want.
    names = [n for n in z.namelist() if n.lower().endswith(".csv")]
    if not names:
        raise RuntimeError("No CSV found inside the zip.")

    with z.open(names[0]) as f:
        return TextIOWrapper(f, encoding="latin1").read()


# ---------------------------------------------------------------------------
# Parsing helpers: extract monthly table and normalize dates
# ---------------------------------------------------------------------------

def _extract_monthly_table(csv_text: str) -> pd.DataFrame:
    """
    Extract the monthly data block from a raw Fama–French CSV text.

    Strategy:
        1. Find the first row that looks like "YYYYMM,<...>" — this marks the
           start of the monthly data.
        2. Take the *last* comma-separated line before that as the header.
        3. Collect contiguous "YYYYMM,<...>" rows until the pattern breaks.
        4. Read that tiny CSV chunk with pandas.

    Returns:
        A pandas DataFrame with the monthly data block.
    """
    from io import StringIO

    lines = csv_text.splitlines()

    # Regex pattern for rows starting with YYYYMM, possibly with leading spaces.
    yyyy_mm = re.compile(r"^\s*\d{6}\s*,")

    # Step 1: locate the first data row
    first_data_idx = None
    for i, line in enumerate(lines):
        if yyyy_mm.match(line):
            first_data_idx = i
            break

    if first_data_idx is None:
        raise RuntimeError("Couldn't find any YYYYMM data row in the CSV text.")

    # Step 2: search upwards for a header line that looks comma-separated
    header_idx = None
    for j in range(first_data_idx - 1, -1, -1):
        if "," in lines[j]:
            header_idx = j
            break

    if header_idx is None:
        raise RuntimeError("Couldn't locate a header line before the data block.")

    # Step 3: collect header + all contiguous YYYYMM rows
    data_lines = [lines[header_idx]]  # header first
    for line in lines[first_data_idx:]:
        if yyyy_mm.match(line):
            data_lines.append(line)
        else:
            # We stop at the first line that breaks the YYYYMM pattern
            break

    # Step 4: read the small CSV chunk into a DataFrame
    tbl_text = "\n".join(data_lines)
    return pd.read_csv(StringIO(tbl_text))


def _to_month_period(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the first column of a DataFrame to a monthly PeriodIndex and
    coerce all other columns to numeric, dropping empty rows/cols.

    Assumes:
        - First column contains year-month codes like 192607, 199912, etc.
    """
    date_col = df.columns[0]

    # Convert date column to pandas PeriodIndex at monthly frequency
    df[date_col] = pd.PeriodIndex(df[date_col].astype(str), freq="M")
    df = df.set_index(date_col)

    # Convert all remaining columns to numeric (invalid values → NaN)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows and columns that are completely empty
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    return df


# ---------------------------------------------------------------------------
# Formatting helper for "pretty" aligned CSV export
# ---------------------------------------------------------------------------

def fmt_num(val, max_decimals: int = 6, dec_char: str = ".") -> str:
    """
    Format a numeric value as a string with a maximum number of decimals
    and no unnecessary trailing zeros.

    Args:
        val: The value to format (can be float or NaN).
        max_decimals: Maximum number of decimal places to show.
        dec_char: Decimal separator to use (e.g. ',' for European style).

    Returns:
        A string representation suitable for aligned CSV export.
    """
    if pd.isna(val):
        return ""

    # Start with a fixed decimal representation (e.g. "0.123400")
    s = f"{val:.{max_decimals}f}"

    # Remove trailing zeros and a possible trailing decimal point
    if "." in s:
        s = s.rstrip("0").rstrip(".")

    # Replace decimal separator if requested
    if dec_char != ".":
        s = s.replace(".", dec_char)

    return s


# ---------------------------------------------------------------------------
# Public functions: download + clean size deciles and FF3
# ---------------------------------------------------------------------------

def download_size_deciles() -> pd.DataFrame:
    """
    Download and clean the 'Portfolios Formed on ME' table.

    We keep only the 10 *decile* portfolios:
        - "Lo 10", "2-Dec", ..., "9-Dec", "Hi 10"

    Then we:
        - rename them to ME1..ME10 (ME1=Lo 10, ME10=Hi 10)
        - convert the index to a monthly PeriodIndex
        - convert returns to numeric
        - replace Fama–French missing flag -99.99 with NaN

    Returns:
        A DataFrame indexed by monthly Period, with columns ME1..ME10.
        Values are still in *percent* (not decimals yet).
    """
    z = _download_zip(URL_SIZE_DECILES)
    csv_text = _read_csv_text(z)
    raw_tbl = _extract_monthly_table(csv_text)

    # Save full raw CSV text for reproducibility / inspection
    (DATA_RAW / "Portfolios_Formed_on_ME_raw.csv").write_text(
        csv_text, encoding="latin1"
    )

    # Normalize column names (strip spaces) for robust matching
    cols = [str(c).strip() for c in raw_tbl.columns]
    raw_tbl.columns = cols

    # Target labels in the original Fama–French header
    decile_labels = ["Lo 10"] + [f"{i}-Dec" for i in range(2, 10)] + ["Hi 10"]

    # Find the actual column names that correspond to those labels
    wanted = []
    for lab in decile_labels:
        # We allow some flexibility (extra words/spaces) by searching as a word
        match = next(
            (
                c
                for c in cols
                if re.search(rf"\b{re.escape(lab)}\b", c, flags=re.IGNORECASE)
            ),
            None,
        )
        if match is None:
            raise RuntimeError(
                f"Couldn't find decile column '{lab}' in headers: {cols[:15]} ..."
            )
        wanted.append(match)

    # Keep only the date column + the 10 decile columns
    date_col = raw_tbl.columns[0]
    keep_cols = [date_col] + wanted
    df = raw_tbl[keep_cols].copy()

    # Convert dates to monthly PeriodIndex and returns to numeric
    df[date_col] = pd.PeriodIndex(df[date_col].astype(str), freq="M")
    df = df.set_index(date_col)
    for c in wanted:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Build mapping to ME1..ME10
    rename_map = {wanted[0]: "ME1"}  # Lo 10 -> ME1
    for i, c in enumerate(wanted[1:-1], start=2):  # 2-Dec..9-Dec -> ME2..ME9
        rename_map[c] = f"ME{i}"
    rename_map[wanted[-1]] = "ME10"  # Hi 10 -> ME10

    df = df.rename(columns=rename_map)

    # French uses -99.99 as a missing-data flag; convert to NaN and drop
    for c in df.columns:
        df[c] = df[c].replace(-99.99, pd.NA)
    df = df.dropna(how="all")

    return df


def download_ff3() -> pd.DataFrame:
    """
    Download and clean the Fama–French 3-factor (plus RF) monthly data.

    We:
        - Extract the monthly data block from the raw CSV
        - Convert dates to a monthly PeriodIndex
        - Keep only the standard FF3 factors: Mkt-RF, SMB, HML, RF

    Returns:
        A DataFrame indexed by monthly Period, with columns "Mkt-RF", "SMB", "HML", "RF".
        Values are still in *percent* (not decimals yet).
    """
    z = _download_zip(URL_FF3)
    csv_text = _read_csv_text(z)
    raw_tbl = _extract_monthly_table(csv_text)
    dfm = _to_month_period(raw_tbl)

    # Save full raw CSV text for reproducibility / inspection
    (DATA_RAW / "F-F_Research_Data_Factors_raw.csv").write_text(
        csv_text, encoding="latin1"
    )

    # Some vintage versions might be missing some columns, so we filter gracefully
    keep = [c for c in ["Mkt-RF", "SMB", "HML", "RF"] if c in dfm.columns]
    dfm = dfm[keep]
    return dfm


# ---------------------------------------------------------------------------
# Script entry-point: merge, convert to decimals, and save pretty CSV
# ---------------------------------------------------------------------------

def main():
    """
    Download size-decile portfolios and FF3 factors, merge them,
    convert returns from percent to decimal, and save a visually aligned CSV.

    The output file is written to OUT (see global constant).
    """
    # ---- Download size deciles ----
    print("Downloading Fama–French size deciles (monthly)…")
    me = download_size_deciles()
    print(
        f"  -> fetched size deciles  ({me.shape[0]} months, {me.shape[1]} cols); "
        "raw saved in data/raw"
    )

    # ---- Download FF3 factors ----
    print("Downloading Fama–French 3 Factors (monthly)…")
    ff3 = download_ff3()
    print(
        f"  -> fetched FF3 factors  ({ff3.shape[0]} months, {ff3.shape[1]} cols); "
        "raw saved in data/raw"
    )

    # ---- Merge on the monthly index ----
    joined = me.join(ff3, how="inner")
    print(
        f"  -> merged table in memory  ({joined.shape[0]} rows, "
        f"{joined.shape[1]} cols, still in PCT)"
    )

    # We now replicate the logic of prepare_data.py:
    # - Ensure monthly index
    # - Convert from percent to decimal
    # - Drop stray unnamed columns
    # - Write a visually aligned, semicolon-separated CSV

    df = joined.copy()

    # Robustly parse the index as a monthly PeriodIndex.
    # If it fails, fall back to a DatetimeIndex.
    try:
        df.index = pd.PeriodIndex(df.index.astype(str), freq="M")
    except Exception:
        df.index = pd.to_datetime(
            df.index.astype(str), format="%Y-%m", errors="coerce"
        )

    # Convert all columns to numeric and divide by 100 to go from percent to decimal.
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0
    df.index.name = "date"

    # Drop any stray "Unnamed:*" columns that sometimes appear in CSV exports.
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # ---- Build a text table for visually aligned CSV output ----

    # Work on a copy with the index turned into a column
    df_txt = df.reset_index().copy()

    # Format all numeric columns using fmt_num; leave the 'date' column as-is.
    for c in df_txt.columns:
        if c != "date":
            df_txt[c] = df_txt[c].map(lambda v: fmt_num(v))

    # Compute column widths as the max between header length and all value lengths
    widths = {
        c: max(len(c), df_txt[c].astype(str).map(len).max())
        for c in df_txt.columns
    }

    # Build header row: date left-aligned, numeric columns right-aligned
    header_cells = [f"{'date':<{widths['date']}}"] + [
        f"{c:>{widths[c]}}" for c in df_txt.columns if c != "date"
    ]

    # Build all data rows
    row_lines = []
    for _, row in df_txt.iterrows():
        left = f"{str(row['date']):<{widths['date']}}"
        nums = [f"{str(row[c]):>{widths[c]}}" for c in df_txt.columns if c != "date"]
        row_lines.append([left] + nums)

    # Ensure output folder exists
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Join cells using semicolons and write to disk (UTF-8 with BOM)
    lines_csv = [";".join(header_cells)] + [";".join(cells) for cells in row_lines]
    OUT.write_text("\n".join(lines_csv), encoding="utf-8-sig")
    print("Saved visually aligned decimal CSV →", OUT)


if __name__ == "__main__":
    main()
