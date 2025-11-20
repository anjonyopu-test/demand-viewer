# demand_viewer_app_v6.py (v6_final11)
# コープさっぽろ｜デマンド値ビューア（固定スキーマ・高速対応・比較ツール）
# - 比較: BがAに対してどれだけ変化?（差分・増減率）
# - 時系列A vs B: 相対日でオーバーレイ（重ね描画）
# - 時刻別形状比較は外し、日付軸の比較を主軸
# - サイドバー期間と比較ツールは独立
# - 既知バグ修正: normalize_date_series 内の括弧タイプミス

import streamlit as st
import pandas as pd
import numpy as np
import re, calendar
from datetime import date, datetime, time as dtime
from pathlib import Path

st.set_page_config(page_title="デマンド値ビューア（高速＋比較）", layout="wide")
st.title("コープさっぽろ｜デマンド値ビューア（高速＋比較）")
st.caption("Parquet高速・Excel最適化／合算ツール／A-B比較ツール（比較は全期間で独立）")

# ===== ユーティリティ =====
def month_end(y: int, m: int) -> date:
    return date(y, m, calendar.monthrange(y, m)[1])

def parse_year_month_from_sheet(sheet_name: str):
    m = re.search(r"(\d{4})年\s*(\d{1,2})月", str(sheet_name))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def sheet_overlaps_period(y: int, m: int, start_date: date, end_date: date) -> bool:
    s = date(y, m, 1)
    e = month_end(y, m)
    return not (e < start_date or s > end_date)

def normalize_date_series(s: pd.Series) -> pd.Series:
    """
    YYYYMMDD / Excel Serial / 明示フォーマット(%Y-%m-%d, %Y/%m/%d, %Y.%m.%d) を許容。
    すべて datetime64[ns] 正規化（00:00固定）まで行う。
    """
    s_str = s.astype(str).str.strip()

    # 1) YYYYMMDD（8桁）
    mask_ymd = s_str.str.fullmatch(r"\d{8}")
    dt1 = pd.to_datetime(s_str.where(mask_ymd, np.nan), format="%Y%m%d", errors="coerce")

    # 2) Excel Serial（数値）
    s_num = pd.to_numeric(s_str.where(~mask_ymd, np.nan), errors="coerce")
    mask_serial = s_num.notna() & (s_num > 0)
    dt2 = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if mask_serial.any():
        dt2.loc[mask_serial] = pd.to_datetime(
            s_num.loc[mask_serial], unit="D", origin="1899-12-30", errors="coerce"
        )

    # 3) 明示フォーマット
    leftover_idx = s.index[dt1.isna() & dt2.isna()]
    dt3 = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if len(leftover_idx) > 0:
        rem = s_str.loc[leftover_idx]
        tried = pd.Series(pd.NaT, index=leftover_idx, dtype="datetime64[ns]")  # ← 修正: 括弧
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
            mask_try = tried.isna()
            if mask_try.any():
                cand = pd.to_datetime(rem[mask_try], format=fmt, errors="coerce")
                tried.loc[mask_try] = cand
        dt3 = tried

    dt = dt1.fillna(dt2).fillna(dt3)
    return pd.to_datetime(dt, errors="coerce").dt.normalize()

def to_long_after_prefilter(df: pd.DataFrame, time_cols: list[str]) -> pd.DataFrame:
    long = df.melt(
        id_vars=["店舗名","日付"],
        value_vars=time_cols,
        var_name="時間帯",
        value_name="デマンド値"
    ).dropna(subset=["店舗名","日付","デマンド値"], how="any")

    long["時間帯"] = long["時間帯"].astype(str).str.replace("\u3000", " ").str.strip()
    long["開始時刻"] = np.where(
        long["時間帯"].str.contains("-"),
        long["時間帯"].str.split("-").str[0].str.strip(),
        long["時間帯"]
    )
    long["日時"] = pd.to_datetime(long["日付"].dt.date.astype(str) + " " + long["開始時刻"], errors="coerce")
    return long

def type_optimize_long(df: pd.DataFrame) -> pd.DataFrame:
    if "店舗名" in df.columns:
        df["店舗名"] = df["店舗名"].astype("category")
    for c in ["時間帯","開始時刻"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    if "デマンド値" in df.columns:
        df["デマンド値"] = pd.to_numeric(df["デマンド値"], errors="coerce").astype("float32")
    if "日時" in df.columns:
        df["日時"] = pd.to_datetime(df["日時"], errors="coerce")
    if "日付" in df.columns:
        df["日付"] = pd.to_datetime(df["日付"], errors="coerce")
    return df

# ===== Excel 読み込み（期間フィルタ付き）=====
@st.cache_data(show_spinner=True, ttl=1800, max_entries=8)
def load_excel_filtered(path_or_file, start_date: date, end_date: date, early_store_filter: list[str] | None):
    xls = pd.ExcelFile(path_or_file)
    frames = []
    diags = []
    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)

    for sheet in xls.sheet_names:
        ym = parse_year_month_from_sheet(sheet)
        if ym:
            y, m = ym
            if not sheet_overlaps_period(y, m, start_date, end_date):
                continue

        df = pd.read_excel(xls, sheet_name=sheet, header=0)
        if df.shape[1] < 3:
            continue

        cols = list(df.columns)
        store_col, date_col = cols[0], cols[1]
        time_cols = [c for c in cols[2:] if df[c].notna().any()]
        if not time_cols:
            continue

        df = df.rename(columns={store_col: "店舗名", date_col: "日付"})
        df["日付"] = normalize_date_series(df["日付"])

        # 期間フィルタ
        mdate = (df["日付"] >= start_ts) & (df["日付"] <= end_ts)
        df = df.loc[mdate]

        # 店舗フィルタ
        if early_store_filter:
            df = df[df["店舗名"].isin(early_store_filter)]

        if df.empty:
            continue

        df[time_cols] = df[time_cols].apply(pd.to_numeric, errors="coerce").astype("float32")

        long = to_long_after_prefilter(df[["店舗名","日付"] + time_cols], time_cols)
        long = type_optimize_long(long)
        frames.append(long)

        diags.append({
            "sheet": sheet,
            "first_cols": cols[:8],
            "n_time_cols_used": len(time_cols),
            "n_rows": int(len(long)),
        })

    if not frames:
        return pd.DataFrame(columns=["店舗名","日付","時間帯","デマンド値","開始時刻","日時"]), diags

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["店舗名","日付","デマンド値"])
    return data, diags

# ===== Parquet 高速 =====
def parquet_dir_path(base_dir: Path) -> Path:
    return base_dir / "data_parquet"

def parquet_available(parquet_dir: Path) -> bool:
    return parquet_dir.exists() and any(parquet_dir.glob("*.parquet"))

@st.cache_data(show_spinner=True, ttl=3600, max_entries=16)
def load_parquet_fast(parquet_dir: Path, start_date: date, end_date: date, early_store_filter: list[str] | None):
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=["店舗名","日付","時間帯","デマンド値","開始時刻","日時"])

    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
    pick = []
    for f in files:
        m = re.search(r"(\d{4})-(\d{2})\.parquet", f.name)
        if m:
            y, mm = int(m.group(1)), int(m.group(2))
            if sheet_overlaps_period(y, mm, start_date, end_date):
                pick.append(f)
        else:
            pick.append(f)

    dfs = []
    for f in pick:
        df = pd.read_parquet(f)
        dcol = pd.to_datetime(df["日付"], errors="coerce")
        m1 = (dcol >= start_ts) & (dcol <= end_ts)
        if early_store_filter:
            m2 = df["店舗名"].isin(early_store_filter)
            df = df.loc[m1 & m2]
        else:
            df = df.loc[m1]
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["店舗名","日付","時間帯","デマンド値","開始時刻","日時"])

    out = pd.concat(dfs, ignore_index=True)
    out = type_optimize_long(out)
    return out

# ===== Parquet 前処理（Excel→Parquet）=====
@st.cache_data(show_spinner=True, ttl=0)
def preconvert_to_parquet(path_or_file, outdir: Path):
    xls = pd.ExcelFile(path_or_file)
    outdir.mkdir(exist_ok=True)
    results = []

    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, header=0)
        if df.shape[1] < 3:
            continue
        cols = list(df.columns)
        df = df.rename(columns={cols[0]: "店舗名", cols[1]: "日付"})
        time_cols = [c for c in cols[2:] if df[c].notna().any()]
        if not time_cols:
            continue

        df["日付"] = normalize_date_series(df["日付"])
        df[time_cols] = df[time_cols].apply(pd.to_numeric, errors="coerce").astype("float32")
        df["店舗名"] = df["店舗名"].astype("category")

        long = to_long_after_prefilter(df[["店舗名","日付"] + time_cols], time_cols)
        long = type_optimize_long(long)

        m = parse_year_month_from_sheet(sheet)
        if m:
            y, mm = m
            fname = f"{y}-{mm:02d}.parquet"
        else:
            safe = re.sub(r"[^\w\-]+", "_", str(sheet))
            fname = f"{safe}.parquet"

        long.to_parquet(outdir / fname, index=False)
        results.append({"sheet": sheet, "rows": int(len(long)), "file": fname})

    return results

# ===== 期間デフォルト推定 =====
def guess_period_from_sources():
    min_guess = date(2000,1,1)
    max_guess = date(2030,12,31)
    if parquet_available(pq_dir):
        files = sorted(pq_dir.glob("*.parquet"))
        yms = []
        for f in files:
            m = re.search(r"(\d{4})-(\d{2})\.parquet", f.name)
            if m:
                yms.append((int(m.group(1)), int(m.group(2))))
        if yms:
            y0, m0 = yms[0]; y1, m1 = yms[-1]
            return date(y0, m0, 1), month_end(y1, m1)
    elif file is not None:
        try:
            xls_tmp = pd.ExcelFile(file)
            yms = [parse_year_month_from_sheet(sh) for sh in xls_tmp.sheet_names]
            yms = [t for t in yms if t]
            if yms:
                y0, m0 = yms[0]; y1, m1 = yms[-1]
                return date(y0, m0, 1), month_end(y1, m1)
        except Exception:
            pass
    return min_guess, max_guess

def guess_full_period_for_compare():
    return guess_period_from_sources()

# ===== サイドバー =====
st.sidebar.header("① データ読み込み（固定スキーマ）")
file = st.sidebar.file_uploader("Excel（.xlsx）を選択（A=店舗名, B=日付, C..=時間帯）", type=["xlsx"])

base_dir = Path.cwd()
pq_dir = parquet_dir_path(base_dir)

st.sidebar.header("高速モード")
use_fast = st.sidebar.toggle("Parquet（前処理済み）を優先して使う", value=True)

if file is not None:
    if st.sidebar.button("⚡ 前処理してParquet生成（高速化A）"):
        with st.spinner("前処理中（Excel→月別Parquet）..."):
            try:
                res = preconvert_to_parquet(file, pq_dir)
                st.sidebar.success(f"Parquet作成: {len(res)} ファイル")
            except Exception as e:
                st.sidebar.error("前処理エラー")
                st.sidebar.exception(e)

# ===== サイドバー 期間 =====
min_guess, max_guess = guess_period_from_sources()

st.sidebar.header("② フィルター")
if st.sidebar.button("🔄 フィルター初期化（セッション）"):
    st.cache_data.clear()
    st.session_state.clear()
    st.rerun()

date_range = st.sidebar.date_input(
    "対象期間（開始日〜終了日）",
    value=(min_guess, max_guess),
    min_value=min_guess,
    max_value=max_guess,
    key="target_period_v6_final11"
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_guess, max_guess

# 店舗候補の推定
store_candidates = []
try:
    if use_fast and parquet_available(pq_dir):
        tmp = load_parquet_fast(pq_dir, start_date, end_date, early_store_filter=None)
        store_candidates = (
            pd.Series(tmp["店舗名"].dropna().astype(str).unique())
            .loc[lambda s: s.ne("") & ~s.str.fullmatch(r"\d+(\.\d+)?")]
            .sort_values()
            .tolist()
        )
    elif file is not None:
        xls = pd.ExcelFile(file)
        names = []
        for sh in xls.sheet_names:
            ym = parse_year_month_from_sheet(sh)
            if ym and not sheet_overlaps_period(ym[0], ym[1], start_date, end_date):
                continue
            small = pd.read_excel(xls, sheet_name=sh, usecols=[0,1], header=0)
            if small.shape[1] >= 2:
                dd = small.copy()
                dd.columns = ["店舗名","日付"]
                dd["日付"] = normalize_date_series(dd["日付"])
                dd = dd.dropna(subset=["日付"])
                vals = (
                    dd["店舗名"].dropna().astype(str).str.strip()
                    .loc[lambda s: s.ne("") & ~s.str.fullmatch(r"\d+(\.\d+)?")]
                    .unique().tolist()
                )
                names.extend(vals)
        store_candidates = sorted(set(names))
except Exception:
    pass

selected_stores = st.sidebar.multiselect(
    "店舗名（複数選択可）",
    store_candidates,
    default=(store_candidates[:1] if store_candidates else [])
)

time_slots_all = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0,30)]
start_time = st.sidebar.selectbox("開始時刻（表示フィルタ）", time_slots_all, index=0)
end_time   = st.sidebar.selectbox("終了時刻（表示フィルタ・含む）", time_slots_all, index=len(time_slots_all)-1)

agg_level = st.sidebar.selectbox("集計粒度", ["30分（そのまま）", "時間別（合計）", "日別（合計）"])

# 合算トグル
if "show_sum" not in st.session_state:
    st.session_state["show_sum"] = False
c1, c2 = st.sidebar.columns(2)
if c1.button("🔢 合算を表示/更新（フィルタ全体）"):
    st.session_state["show_sum"] = True
if c2.button("🙈 合算を隠す"):
    st.session_state["show_sum"] = False

# 比較トグル（独立）
if "show_compare" not in st.session_state:
    st.session_state["show_compare"] = False
cc1, cc2 = st.sidebar.columns(2)
if cc1.button("🔍 比較ツールを開く"):
    st.session_state["show_compare"] = True
if cc2.button("✖ 比較ツールを閉じる"):
    st.session_state["show_compare"] = False

# ===== メインデータ読み込み（サイドバー期間）=====
if use_fast and parquet_available(pq_dir):
    with st.spinner("Parquet高速読込中..."):
        data = load_parquet_fast(pq_dir, start_date, end_date, early_store_filter=selected_stores or None)
        diags = []
else:
    if file is None:
        st.info("左のサイドバーからExcelをアップロードするか、Parquet高速モードを使ってください。")
        st.stop()
    with st.spinner("Excel 読込・最適化中..."):
        data, diags = load_excel_filtered(file, start_date, end_date, early_store_filter=selected_stores or None)

with st.expander("読み込み診断情報（先頭10件）"):
    st.write(diags[:10] if diags else "Parquet高速モード")

if data.empty:
    st.warning("条件に合致するデータがありません。")
    st.stop()

# 時刻フィルタ
data["開始時刻_only"] = pd.to_datetime(data["日時"], errors="coerce").dt.strftime("%H:%M")
mask_time = (data["開始時刻_only"] >= start_time) & (data["開始時刻_only"] <= end_time)
filtered = data.loc[mask_time].copy()
if filtered.empty:
    st.warning("条件に合致するデータがありません。")
    st.stop()

# 可視化用集計
if agg_level == "30分（そのまま）":
    show = filtered.sort_values(["店舗名","日時"])
    xcol = "日時"
elif agg_level == "時間別（合計）":
    filtered["hour"] = pd.to_datetime(filtered["日時"], errors="coerce").dt.floor("H")
    show = (
        filtered.groupby(["店舗名","hour"], as_index=False)["デマンド値"]
        .sum().rename(columns={"hour":"日時"})
    )
    xcol = "日時"
else:
    filtered["日付_d"] = pd.to_datetime(filtered["日付"], errors="coerce").dt.date
    show = (
        filtered.groupby(["店舗名","日付_d"], as_index=False)["デマンド値"]
        .sum().rename(columns={"日付_d":"日付"})
    )
    xcol = "日付"

# 重複列抑止
cols_drop = []
if "日付" in show.columns and "日時" in show.columns:
    if "日時" == xcol:
        cols_drop.append("日付")
    else:
        cols_drop.append("日時")
for c in ["開始時刻_only"]:
    if c in show.columns: cols_drop.append(c)
if cols_drop:
    show = show.drop(columns=cols_drop)

# ===== 合算セクション =====
st.subheader("集計セクション（合計ツール）")
if not st.session_state.get("show_sum", False):
    st.info("サイドバーの **「🔢 合算を表示/更新（フィルタ全体）」** を押すと展開されます。")
else:
    total_sum = float(filtered["デマンド値"].sum())
    k1, k2, k3 = st.columns(3)
    k1.metric("総合計（表示中の時間帯条件）", f"{total_sum:,.2f}")
    by_store = (
        filtered.groupby("店舗名", as_index=False)["デマンド値"]
        .sum().sort_values("デマンド値", ascending=False)
    )
    k2.metric("店舗数（選択）", f"{by_store['店舗名'].nunique():,}")
    k3.metric("レコード数（選択）", f"{len(filtered):,}")

    with st.expander("店舗別 合計（ダウンロード可）", expanded=True):
        st.dataframe(by_store, use_container_width=True)
        st.download_button(
            "店舗別_合計.csv をダウンロード",
            data=by_store.to_csv(index=False).encode("utf-8-sig"),
            file_name="店舗別_合計.csv",
            mime="text/csv",
        )

    # 選択行合算（チェックボックス）
    st.markdown("### 🧮 選択行の合算（表からチェック）")
    if "table_rev_v6" not in st.session_state:
        st.session_state["table_rev_v6"] = 0
    editor_key = f"table_editor_v6_final11_{st.session_state['table_rev_v6']}"
    table_df = show.reset_index(drop=True).copy()
    if "選択" not in table_df.columns:
        table_df.insert(0, "選択", False)
    disabled_cols = [c for c in table_df.columns if c != "選択"]

    edited = st.data_editor(
        table_df,
        use_container_width=True,
        num_rows="fixed",
        column_config={"選択": st.column_config.CheckboxColumn("選択", help="合算したい行にチェック")},
        disabled=disabled_cols,
        key=editor_key,
    )

    cc1, cc2 = st.columns([1,1])
    if "show_sum_selected" not in st.session_state:
        st.session_state["show_sum_selected"] = False
    if cc1.button("🧮 合算（選択行）を表示/更新"):
        st.session_state["show_sum_selected"] = True
    if cc2.button("🧹 選択をクリア"):
        st.session_state["table_rev_v6"] += 1
        st.session_state["show_sum_selected"] = False
        st.rerun()

    if st.session_state.get("show_sum_selected", False):
        sel = edited[edited["選択"] == True].copy()
        if sel.empty:
            st.info("チェックされた行がありません。")
        else:
            total_sel = float(sel["デマンド値"].sum())
            s1, s2 = st.columns(2)
            s1.metric("選択行の合計", f"{total_sel:,.2f}")
            if "店舗名" in sel.columns:
                by_store_sel = sel.groupby("店舗名", as_index=False)["デマンド値"].sum().sort_values("デマンド値", ascending=False)
                with st.expander("選択行：店舗別 合計", expanded=True):
                    st.dataframe(by_store_sel, use_container_width=True)

    # 任意期間 合算（サイドバー時間帯無視）
    st.markdown("### ⏱ 任意の開始日時〜終了日時で合算（サイドバー時間帯は無視）")
    min_dt = datetime.combine(start_date, dtime(0,0))
    max_dt = datetime.combine(end_date, dtime(23,30))
    col_a, col_b = st.columns(2)
    dt_start = col_a.date_input("開始日（合算用）", value=min_dt.date(), min_value=start_date, max_value=end_date, key="sum_custom_start_date")
    tm_start = col_a.time_input("開始時刻（合算用）", value=dtime(0,0), step=1800, key="sum_custom_start_time")
    dt_end   = col_b.date_input("終了日（合算用）", value=max_dt.date(), min_value=start_date, max_value=end_date, key="sum_custom_end_date")
    tm_end   = col_b.time_input("終了時刻（合算用）", value=dtime(23,30), step=1800, key="sum_custom_end_time")

    if st.button("⏱ この開始日時〜終了日時で合算する"):
        try:
            start_dt = datetime.combine(dt_start, tm_start)
            end_dt   = datetime.combine(dt_end, tm_end)
            if start_dt > end_dt:
                st.error("開始日時は終了日時以前で指定してください。")
            else:
                base = data.copy()
                base["日時"] = pd.to_datetime(base["日時"], errors="coerce")
                m = (base["日時"] >= pd.Timestamp(start_dt)) & (base["日時"] <= pd.Timestamp(end_dt))
                base = base.loc[m]
                if base.empty:
                    st.info("指定範囲に合致するデータがありません。")
                else:
                    total_custom = float(base["デマンド値"].sum())
                    c1, c2 = st.columns(2)
                    c1.metric("任意期間の合計", f"{total_custom:,.2f}")
                    by_store_custom = base.groupby("店舗名", as_index=False)["デマンド値"].sum().sort_values("デマンド値", ascending=False)
                    with st.expander("任意期間：店舗別 合計（ダウンロード可）", expanded=True):
                        st.dataframe(by_store_custom, use_container_width=True)
                        st.download_button(
                            "任意期間_店舗別_合計.csv をダウンロード",
                            data=by_store_custom.to_csv(index=False).encode("utf-8-sig"),
                            file_name="任意期間_店舗別_合計.csv",
                            mime="text/csv",
                        )
        except Exception as e:
            st.error("任意期間合算でエラーが発生しました。")
            st.exception(e)

# ===== 比較ツール（全期間・独立）=====
if st.session_state.get("show_compare", False):
    st.subheader("🔍 比較ツール（店舗1つ・A期間 vs B期間｜全期間独立）")

    min_all, max_all = guess_full_period_for_compare()

    @st.cache_data(show_spinner=True, ttl=1800)
    def load_compare_data(file, pq_dir, use_fast, min_all, max_all):
        if use_fast and parquet_available(pq_dir):
            return load_parquet_fast(pq_dir, min_all, max_all, early_store_filter=None)
        else:
            if file is None:
                return pd.DataFrame(columns=["店舗名","日付","時間帯","デマンド値","開始時刻","日時"])
            data_all, _ = load_excel_filtered(file, min_all, max_all, early_store_filter=None)
            return data_all

    with st.spinner("比較用データ（全期間）を準備中..."):
        cmp_data = load_compare_data(file, pq_dir, use_fast, min_all, max_all)

    if cmp_data.empty:
        st.info("比較対象データが見つかりません（全期間）。")
    else:
        compare_stores = (
            pd.Series(cmp_data["店舗名"].dropna().astype(str).unique())
            .loc[lambda s: s.ne("") & ~s.str.fullmatch(r"\d+(\.\d+)?")]
            .sort_values().tolist()
        )
        csel = st.selectbox("店舗（比較対象）", compare_stores, index=0, key="compare_store_v6_11")

        base_min_date = pd.to_datetime(cmp_data["日付"], errors="coerce").min().date()
        base_max_date = pd.to_datetime(cmp_data["日付"], errors="coerce").max().date()
        time_all = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0,30)]

        st.markdown("#### 期間A")
        ca1, ca2 = st.columns([1,1])
        a_start_d = ca1.date_input("A 開始日", value=base_min_date, min_value=base_min_date, max_value=base_max_date, key="cmp_A_sd_11")
        a_end_d   = ca1.date_input("A 終了日", value=base_max_date, min_value=base_min_date, max_value=base_max_date, key="cmp_A_ed_11")
        a_start_t = ca2.selectbox("A 開始時刻", time_all, index=0, key="cmp_A_st_11")
        a_end_t   = ca2.selectbox("A 終了時刻（含む）", time_all, index=len(time_all)-1, key="cmp_A_et_11")

        st.markdown("#### 期間B")
        cb1, cb2 = st.columns([1,1])
        b_start_d = cb1.date_input("B 開始日", value=base_min_date, min_value=base_min_date, max_value=base_max_date, key="cmp_B_sd_11")
        b_end_d   = cb1.date_input("B 終了日", value=base_max_date, min_value=base_min_date, max_value=base_max_date, key="cmp_B_ed_11")
        b_start_t = cb2.selectbox("B 開始時刻", time_all, index=0, key="cmp_B_st_11")
        b_end_t   = cb2.selectbox("B 終了時刻（含む）", time_all, index=len(time_all)-1, key="cmp_B_et_11")

        run_compare = st.button("🆚 この設定で比較する（BがAからどれだけ変化？）")
        if run_compare:
            try:
                a_start_dt = pd.Timestamp(f"{a_start_d} {a_start_t}:00")
                a_end_dt   = pd.Timestamp(f"{a_end_d} {a_end_t}:00")
                b_start_dt = pd.Timestamp(f"{b_start_d} {b_start_t}:00")
                b_end_dt   = pd.Timestamp(f"{b_end_d} {b_end_t}:00")

                if a_start_dt > a_end_dt or b_start_dt > b_end_dt:
                    st.error("開始日時は終了日時以前で指定してください。")
                else:
                    base = cmp_data.copy()
                    base["日時"] = pd.to_datetime(base["日時"], errors="coerce")
                    base = base[base["店舗名"].astype(str) == str(csel)]

                    A = base[(base["日時"] >= a_start_dt) & (base["日時"] <= a_end_dt)].copy()
                    B = base[(base["日時"] >= b_start_dt) & (base["日時"] <= b_end_dt)].copy()

                    if A.empty or B.empty:
                        st.info("AまたはBの期間にデータがありません。")
                    else:
                        # 全体合計の比較（BがAからどれだけ変化したか）
                        A_sum = float(A["デマンド値"].sum())
                        B_sum = float(B["デマンド値"].sum())
                        diff  = B_sum - A_sum
                        pct   = (diff / A_sum * 100.0) if A_sum != 0 else np.nan

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("A 合計", f"{A_sum:,.2f}")
                        m2.metric("B 合計", f"{B_sum:,.2f}")
                        m3.metric("差分(B−A)", f"{diff:,.2f}")
                        m4.metric("増減率(%)", f"{pct:,.2f}" if np.isfinite(pct) else "N/A")

                        # ===== 日別合計で比較（基本）
                        A["日付_only"] = pd.to_datetime(A["日時"]).dt.date
                        B["日付_only"] = pd.to_datetime(B["日時"]).dt.date
                        Ag = A.groupby("日付_only", as_index=False)["デマンド値"].sum().rename(columns={"デマンド値":"A_合計"})
                        Bg = B.groupby("日付_only", as_index=False)["デマンド値"].sum().rename(columns={"デマンド値":"B_合計"})

                        # 相対日インデックス（1..N）で整列 → 重ね描画
                        Ag = Ag.sort_values("日付_only").reset_index(drop=True).assign(相対日=lambda df: df.index+1)
                        Bg = Bg.sort_values("日付_only").reset_index(drop=True).assign(相対日=lambda df: df.index+1)

                        # 相対日でマージして日毎の変化量
                        comp = pd.merge(Ag, Bg, on="相対日", how="outer")
                        comp["A_合計"] = comp["A_合計"].fillna(0.0)
                        comp["B_合計"] = comp["B_合計"].fillna(0.0)
                        comp["差分(B−A)"] = comp["B_合計"] - comp["A_合計"]
                        comp["増減率(%)"] = np.where(
                            comp["A_合計"] != 0, (comp["差分(B−A)"]/comp["A_合計"])*100.0, np.nan
                        )
                        comp = comp.rename(columns={"日付_only_x":"A_日付", "日付_only_y":"B_日付"})

                        import altair as alt

                        # 1) オーバーレイ（相対日）: A/Bを同じx軸上に重ねる
                        rel_long = pd.concat([
                            comp[["相対日","A_合計"]].rename(columns={"A_合計":"値"}).assign(系列="A"),
                            comp[["相対日","B_合計"]].rename(columns={"B_合計":"値"}).assign(系列="B"),
                        ], ignore_index=True)

                        st.markdown("##### 時系列（オーバーレイ｜相対日）")
                        line_rel = alt.Chart(rel_long).mark_line().encode(
                            x=alt.X("相対日:Q", title="相対日（各期間の開始日を1日目として整列）"),
                            y=alt.Y("値:Q", title="日別合計（デマンド値）"),
                            color="系列:N",
                            tooltip=["系列","相対日","値:Q"]
                        ).properties(height=320)
                        st.altair_chart(line_rel, use_container_width=True)

                        # 2) 実日付での参照（重ならないが実カレンダー基準で確認）
                        real_long = pd.concat([
                            Ag.rename(columns={"A_合計":"値","日付_only":"日付"}).assign(系列="A")[["日付","値","系列"]],
                            Bg.rename(columns={"B_合計":"値","日付_only":"日付"}).assign(系列="B")[["日付","値","系列"]],
                        ], ignore_index=True)

                        with st.expander("🗓 実日付ベースの時系列（参考）"):
                            line_real = alt.Chart(real_long).mark_line().encode(
                                x=alt.X("日付:T", title="日付（実カレンダー）"),
                                y=alt.Y("値:Q", title="日別合計"),
                                color="系列:N",
                                tooltip=["系列","日付:T","値:Q"]
                            ).properties(height=300)
                            st.altair_chart(line_real, use_container_width=True)

                        # 日別の差分テーブル（ダウンロード可）
                        with st.expander("🧾 日別 比較表（BがAからどれだけ変化?）", expanded=True):
                            view = comp[["相対日","A_日付","B_日付","A_合計","B_合計","差分(B−A)","増減率(%)"]]
                            st.dataframe(view, use_container_width=True)
                            st.download_button(
                                f"{csel}_日別_AvsB_比較.csv をダウンロード",
                                data=view.to_csv(index=False).encode("utf-8-sig"),
                                file_name=f"{csel}_日別_AvsB_比較.csv",
                                mime="text/csv",
                            )

            except Exception as e:
                st.error("比較ツールでエラーが発生しました。")
                st.exception(e)

# ===== 可視化（基本ライン）=====
st.subheader("可視化（粒度別）")
import altair as alt
line = alt.Chart(show).mark_line().encode(
    x=alt.X(f"{xcol}:T", title=xcol),
    y=alt.Y("デマンド値:Q"),
    color="店舗名:N",
    tooltip=list(show.columns)
).properties(height=360)
st.altair_chart(line, use_container_width=True)

# ===== 表（ダウンロード可）=====
st.subheader("表（ダウンロード可）")
st.dataframe(show.reset_index(drop=True), use_container_width=True)
st.download_button(
    "（表示中の表）CSVをダウンロード",
    data=show.to_csv(index=False).encode("utf-8-sig"),
    file_name="デマンド値_抽出結果.csv",
    mime="text/csv",
)

# ===== デバッグ =====
with st.expander("🧪 デバッグ"):
    st.write({
        "use_fast": use_fast,
        "parquet_available": parquet_available(pq_dir),
        "parquet_dir": str(pq_dir),
        "sidebar_period": (str(start_date), str(end_date)),
        "n_rows_show": len(show),
        "stores_selected": (selected_stores[:10] if selected_stores else []),
    })
