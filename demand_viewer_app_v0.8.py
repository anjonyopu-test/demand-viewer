# demand_viewer_app_v0.8.py
# コープさっぽろ｜デマンド値ビューア（初心者向けUI版＋気温×デマンドパターン分析）
#
# メニュー構成：
#   - ホーム
#   - デマンド分析ツール
#   - 気温×デマンド分析（相関 / 同条件比較 / パターン分析）

import streamlit as st
import pandas as pd
import numpy as np
import re
import calendar
from datetime import date, datetime, time as dtime
from pathlib import Path
import altair as alt
import io
import requests

# ====================== 設定：Googleドライブ関連（デマンド分析ツール用） ======================
DRIVE_FILE_ID_PARTIAL = "1JggplB9IDXOFKnJnD8mZqPiiYGitxdto"  # 部分データ
DRIVE_FILE_ID_FULL = "1E2T9En5whdGy-CpyXvTkx8_Od4HbVi9u"     # 全量データ

def download_excel_from_gdrive(file_id: str) -> bytes:
    """
    Googleドライブ上の共有ファイルをダウンロードして bytes を返す。
    ※ ファイルは「リンクを知っている全員が閲覧可」にしておく必要あり。
    """
    if not file_id or "ここに" in file_id:
        raise ValueError("GoogleドライブのファイルIDが設定されていません。コード内の DRIVE_FILE_ID_* を確認してください。")

    url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.content  # bytes

SAMPLE_DRIVE_URL = "https://drive.google.com"

# ====================== ページ設定 ======================
st.set_page_config(page_title="デマンド値ビューア", layout="wide")

# ---------------------- メインメニュー ----------------------
st.title("デマンド値ビューア")
st.caption("店舗のデマンド（需要）データを、かんたんに確認・比較できるアプリです。")

col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    menu = st.radio(
        "メインメニュー",
        ["ホーム", "デマンド分析ツール", "気温×デマンド分析"],
        horizontal=True,
    )

# ====================== 공통 유틸 함수들 ======================
def parse_year_month_from_sheet(sheet_name: str):
    m = re.search(r"(\d{4})年\s*(\d{1,2})月", str(sheet_name))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def normalize_date_series(s: pd.Series) -> pd.Series:
    s_str = s.astype(str).str.strip()

    mask_ymd = s_str.str.fullmatch(r"\d{8}")
    dt1 = pd.to_datetime(s_str.where(mask_ymd, np.nan), format="%Y%m%d", errors="coerce")

    s_num = pd.to_numeric(s_str.where(~mask_ymd, np.nan), errors="coerce")
    mask_serial = s_num.notna() & (s_num > 0)
    dt2 = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if mask_serial.any():
        dt2.loc[mask_serial] = pd.to_datetime(
            s_num.loc[mask_serial], unit="D", origin="1899-12-30", errors="coerce"
        )

    leftover_idx = s.index[dt1.isna() & dt2.isna()]
    dt3 = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if len(leftover_idx) > 0:
        rem = s_str.loc[leftover_idx]
        tried = pd.Series(pd.NaT, index=leftover_idx, dtype="datetime64[ns]")
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
        id_vars=["店舗名", "日付"],
        value_vars=time_cols,
        var_name="時間帯",
        value_name="デマンド値"
    ).dropna(subset=["店舗名", "日付", "デマンド値"], how="any")

    long["時間帯"] = (
        long["時間帯"]
        .astype(str)
        .str.replace("\u3000", " ")
        .str.strip()
    )
    long["開始時刻"] = np.where(
        long["時間帯"].str.contains("-"),
        long["時間帯"].str.split("-").str[0].str.strip(),
        long["時間帯"]
    )
    long["日時"] = pd.to_datetime(
        long["日付"].dt.date.astype(str) + " " + long["開始時刻"],
        errors="coerce"
    )
    return long

def type_optimize_long(df: pd.DataFrame) -> pd.DataFrame:
    if "店舗名" in df.columns:
        df["店舗名"] = df["店舗名"].astype("category")
    for c in ["時間帯", "開始時刻"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    if "デマンド値" in df.columns:
        df["デマンド値"] = pd.to_numeric(df["デマンド値"], errors="coerce").astype("float32")
    if "日時" in df.columns:
        df["日時"] = pd.to_datetime(df["日時"], errors="coerce")
    if "日付" in df.columns:
        df["日付"] = pd.to_datetime(df["日付"], errors="coerce")
    return df

def parquet_dir_path(base_dir: Path) -> Path:
    return base_dir / "data_parquet"

def parquet_available(parquet_dir: Path) -> bool:
    return parquet_dir.exists() and any(parquet_dir.glob("*.parquet"))

# ====================== ホーム画面 ======================
if menu == "ホーム":
    with col_center:
        st.markdown(
            """
### このアプリでできること

- 店舗ごとのデマンド値を、**期間・店舗・時間帯**で絞り込み  
- グラフで推移を確認  
- 店舗別の合計値を一覧で確認  
- 1つの店舗について、**期間A と 期間B を比較**して増減をチェック  
- 気温とデマンド値の**相関関係**を時間帯ごとに確認  
- 同じ条件（時間帯・気温帯など）で、2つのパターン（例：通常日 vs 大雪日）を比較  
- 冬期・夏期・中間期・10月に分けた**パターン分析（外れ値除外付き）**

---
"""
        )

        with st.expander("💡 よくある質問（Q&A）"):
            st.markdown(
                """
**Q. どんなファイルを読み込めますか？**  

- デマンド分析ツール  
    → Excel（.xlsx）で、1列目：店舗名、2列目：日付、3列目以降：時間帯（例：`0:00-0:30`）  
- 気温×デマンド分析  
    → 1つのファイルの中に  
      `日付 / 時間帯 / 気温 / デマンド値` が入っているデータを想定しています。

**Q. 動きが重いときは？**  
A. デマンド分析ツールでは、左のサイドバーで「読み込みを速くする（事前に変換したデータを使う）」をオンにしてみてください。
"""
            )

            st.markdown("---")
            st.markdown("**サンプルデータの説明などを載せたい場合**")
            st.caption("必要に応じて、Googleドライブなどの説明ページへのリンクを置くことができます。")

            st.link_button(
                "ℹ️ サンプルデータの説明ページを開く",
                SAMPLE_DRIVE_URL,
                help="クリックすると、ブラウザで説明用ページが開きます。",
            )

        st.info("上のメニューで **「デマンド分析ツール」** か **「気温×デマンド分析」** を選ぶと、該当のツール画面が開きます。")
    st.stop()

# ======================================================================
#  メニュー２：デマンド分析ツール（기존 기능 그대로)
# ======================================================================
if menu == "デマンド分析ツール":

    base_dir = Path.cwd()
    pq_dir = parquet_dir_path(base_dir)

    @st.cache_data(show_spinner=True, ttl=1800, max_entries=8)
    def load_excel_all(path_or_file):
        if hasattr(path_or_file, "seek"):
            try:
                path_or_file.seek(0)
            except Exception:
                pass

        xls = pd.ExcelFile(path_or_file, engine="openpyxl")

        frames = []
        diags = []

        for sheet in xls.sheet_names:
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
            df[time_cols] = df[time_cols].apply(pd.to_numeric, errors="coerce").astype("float32")

            long = to_long_after_prefilter(df[["店舗名", "日付"] + time_cols], time_cols)
            long = type_optimize_long(long)
            frames.append(long)

            diags.append({
                "sheet": sheet,
                "n_time_cols": len(time_cols),
                "n_rows": int(len(long)),
            })

        if not frames:
            return pd.DataFrame(columns=["店舗名", "日付", "時間帯", "デマンド値", "開始時刻", "日時"]), diags

        data = pd.concat(frames, ignore_index=True)
        data = data.dropna(subset=["店舗名", "日付", "デマンド値"])
        return data, diags

    @st.cache_data(show_spinner=True, ttl=3600, max_entries=16)
    def load_parquet_all(parquet_dir: Path):
        files = sorted(parquet_dir.glob("*.parquet"))
        if not files:
            return pd.DataFrame(columns=["店舗名", "日付", "時間帯", "デマンド値", "開始時刻", "日時"]), []

        frames = []
        for f in files:
            df = pd.read_parquet(f)
            df = type_optimize_long(df)
            frames.append(df)

        if not frames:
            return pd.DataFrame(columns=["店舗名", "日付", "時間帯", "デマンド値", "開始時刻", "日時"]), []

        data = pd.concat(frames, ignore_index=True)
        data = data.dropna(subset=["店舗名", "日付", "デマンド値"])
        diags = [{"file": f.name, "n_rows": int(len(pd.read_parquet(f)))} for f in files]
        return data, diags

    @st.cache_data(show_spinner=True, ttl=0)
    def preconvert_to_parquet(path_or_file, outdir: Path):
        if hasattr(path_or_file, "seek"):
            try:
                path_or_file.seek(0)
            except Exception:
                pass

        xls = pd.ExcelFile(path_or_file, engine="openpyxl")

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

            long = to_long_after_prefilter(df[["店舗名", "日付"] + time_cols], time_cols)
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

    # ---------- サイドバー（デマンド分析用） ----------
    st.sidebar.title("操作パネル（デマンド分析）")

    base_dir = Path.cwd()
    pq_dir = parquet_dir_path(base_dir)

    st.sidebar.markdown("### ① データを読み込む")

    uploaded_file = st.sidebar.file_uploader(
        "デマンドデータ（Excel）を選択（手動アップロード）",
        type=["xlsx"],
        help="1列目：店舗名、2列目：日付、3列目以降：時間帯（例：0:00-0:30）の形式を想定しています。",
    )

    st.sidebar.markdown("#### 🔁 Googleドライブから自動で読み込む")

    col_p, col_f = st.sidebar.columns(2)
    if col_p.button("部分データ", use_container_width=True):
        st.session_state["auto_file_source"] = "partial"
    if col_f.button("全量データ", use_container_width=True):
        st.session_state["auto_file_source"] = "full"

    auto_source = st.session_state.get("auto_file_source", None)
    auto_file_obj = None

    if auto_source in ("partial", "full"):
        key_bytes = f"auto_file_bytes_{auto_source}"
        cached_bytes = st.session_state.get(key_bytes, None)

        if cached_bytes is not None:
            auto_file_obj = io.BytesIO(cached_bytes)
            st.sidebar.info("Googleドライブからのファイル（キャッシュ済み）を利用しています。")
        else:
            try:
                with st.spinner("Googleドライブからファイルを取得中です..."):
                    if auto_source == "partial":
                        file_bytes = download_excel_from_gdrive(DRIVE_FILE_ID_PARTIAL)
                        st.sidebar.success("部分データをGoogleドライブから読み込みました。")
                    else:
                        file_bytes = download_excel_from_gdrive(DRIVE_FILE_ID_FULL)
                        st.sidebar.success("全量データをGoogleドライブから読み込みました。")

                    st.session_state[key_bytes] = file_bytes
                    auto_file_obj = io.BytesIO(file_bytes)
            except Exception as e:
                st.sidebar.error("Googleドライブからの読み込みに失敗しました。")
                st.sidebar.exception(e)
                auto_file_obj = None

    input_file = auto_file_obj if auto_file_obj is not None else uploaded_file

    st.sidebar.markdown("**読み込みを速くする（任意）**")
    use_fast = st.sidebar.toggle(
        "読み込みを速くする",
        value=True,
        help="事前に変換したデータを使って、読み込み時間を短くします。初回は少し時間がかかります。",
    )

    if input_file is not None:
        if st.sidebar.button("⏩ データを変換して保存（読み込みを速くする）"):
            with st.spinner("変換中です。しばらくお待ちください..."):
                try:
                    res = preconvert_to_parquet(input_file, pq_dir)
                    st.sidebar.success(f"変換が完了しました。（{len(res)}ファイル）")
                except Exception as e:
                    st.sidebar.error("データの変換中にエラーが発生しました。")
                    st.sidebar.exception(e)

    st.sidebar.markdown("---")

    st.sidebar.markdown("### ② 表示の設定")

    agg_level = st.sidebar.selectbox(
        "時間のまとまり方",
        ["30分ごと（元データそのまま）", "1時間ごとに合計", "1日ごとに合計"],
        help="グラフや表で表示するときの時間の粒度を選びます。",
    )

    if st.sidebar.button("🔄 すべての設定とデータをリセット"):
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()

    st.sidebar.markdown("---")

    st.sidebar.markdown("### ③ 追加ツール")

    show_sum = st.sidebar.checkbox(
        "合計・店舗別集計を表示する",
        value=st.session_state.get("show_sum", False),
    )
    st.session_state["show_sum"] = show_sum

    show_compare = st.sidebar.checkbox(
        "1店舗の期間Aと期間Bを比較する",
        value=st.session_state.get("show_compare", False),
    )
    st.session_state["show_compare"] = show_compare

    # ---------- データ読み込み ----------
    if use_fast and parquet_available(pq_dir):
        with st.spinner("データを読み込んでいます（高速モード：Parquet）..."):
            data, diags = load_parquet_all(pq_dir)
    else:
        if input_file is None:
            st.info("左の「① データを読み込む」でファイルを指定するか、Googleドライブから自動読み込みしてください。")
            st.stop()
        with st.spinner("Excelファイルを読み込んでいます..."):
            data, diags = load_excel_all(input_file)

    if data.empty:
        st.warning("有効なデータが読み込めませんでした。ファイルの内容・形式をご確認ください。")
        st.stop()

    data["日時"] = pd.to_datetime(data["日時"], errors="coerce")
    data["日付_only"] = data["日時"].dt.date
    data["時刻_only"] = data["日時"].dt.time

    date_series = pd.to_datetime(data["日付"], errors="coerce")
    valid_dates = date_series.dropna()
    if valid_dates.empty:
        st.error("日付の情報が正しく読み込めていません。Excelの2列目（日付列）をご確認ください。")
        st.stop()

    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()

    store_candidates = (
        pd.Series(data["店舗名"].dropna().astype(str).unique())
        .loc[lambda s: s.ne("") & ~s.str.fullmatch(r"\d+(\.\d+)?")]
        .sort_values()
        .tolist()
    )

    # ---------- ① 기간・店舗 ----------
    st.markdown("## 🔍 分析する条件をえらぶ")

    with st.expander("① 日付と店舗をえらぶ", expanded=True):
        col_d1, col_d2 = st.columns(2)
        start_date = col_d1.date_input(
            "開始日",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
        )
        end_date = col_d2.date_input(
            "終了日",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
        )

        if start_date > end_date:
            st.error("開始日は終了日より前にしてください。")
            st.stop()

        selected_stores = st.multiselect(
            "対象の店舗（1つ以上えらんでください）",
            store_candidates,
            default=[],
            help="店舗をえらぶと、下にグラフや集計結果が表示されます。",
        )

    if not selected_stores:
        analysis_enabled = False
        st.info("店舗を1つ以上えらぶと、グラフや集計結果が表示されます。")
    else:
        analysis_enabled = True

    if analysis_enabled:
        mask_base = (data["日付_only"] >= start_date) & (data["日付_only"] <= end_date)
        mask_base &= data["店舗名"].astype(str).isin(selected_stores)

        base_filtered = data.loc[mask_base].copy()
        if base_filtered.empty:
            st.warning("指定した期間と店舗の組み合わせにデータがありません。条件を少し広げてお試しください。")
            analysis_enabled = False

    # ---------- ② 時間帯 ----------
    if analysis_enabled:
        with st.expander("② 時間帯をえらぶ（必要な場合だけ）", expanded=False):
            time_mode = st.radio(
                "時間のえらび方",
                ["そのまま使う（時間で絞り込まない）", "毎日 同じ時間帯だけ見る", "開始日時〜終了日時でまとめて見る"],
                horizontal=False,
                help=(
                    "・「そのまま使う」：日付と店舗だけで絞り込みます\n"
                    "・「毎日 同じ時間帯だけ見る」：例）6/1〜6/30 の毎日 3:00〜6:00\n"
                    "・「開始日時〜終了日時でまとめて見る」：例）6/1 3:00〜6/30 6:00 までを一気に見る"
                ),
            )

            time_filtered = base_filtered.copy()

            if time_mode == "毎日 同じ時間帯だけ見る":
                c1, c2 = st.columns(2)
                t_start = c1.time_input("毎日の開始時刻", value=dtime(0, 0), step=1800)
                t_end = c2.time_input("毎日の終了時刻", value=dtime(23, 30), step=1800)

                if t_start > t_end:
                    st.error("開始時刻は終了時刻より前にしてください。")
                    st.stop()

                mask_t = (time_filtered["時刻_only"] >= t_start) & (time_filtered["時刻_only"] <= t_end)
                time_filtered = time_filtered.loc[mask_t]

                st.caption(
                    f"📌 {start_date}〜{end_date} の毎日 "
                    f"{t_start.strftime('%H:%M')}〜{t_end.strftime('%H:%M')} にしぼり込み中"
                )

            elif time_mode == "開始日時〜終了日時でまとめて見る":
                c1, c2 = st.columns(2)
                abs_s_date = c1.date_input(
                    "開始日", value=start_date, min_value=min_date, max_value=max_date, key="abs_sd"
                )
                abs_s_time = c1.time_input(
                    "開始時刻", value=dtime(0, 0), step=1800, key="abs_st"
                )
                abs_e_date = c2.date_input(
                    "終了日", value=end_date, min_value=min_date, max_value=max_date, key="abs_ed"
                )
                abs_e_time = c2.time_input(
                    "終了時刻", value=dtime(23, 30), step=1800, key="abs_et"
                )

                abs_start = datetime.combine(abs_s_date, abs_s_time)
                abs_end = datetime.combine(abs_e_date, abs_e_time)
                if abs_start > abs_end:
                    st.error("開始日時は終了時刻より前にしてください。")
                    st.stop()

                mask_t = (time_filtered["日時"] >= abs_start) & (time_filtered["日時"] <= abs_end)
                time_filtered = time_filtered.loc[mask_t]

                st.caption(f"📌 {abs_start} 〜 {abs_end} にしぼり込み中")

        if time_mode == "そのまま使う（時間で絞り込まない）":
            filtered = base_filtered.copy()
        else:
            filtered = time_filtered.copy()

        if filtered.empty:
            st.warning("指定した時間帯にはデータがありません。時間帯の条件を変えてお試しください。")
            analysis_enabled = False

    # ---------- グラフ・集計・表 ----------
    if analysis_enabled:
        if agg_level == "30分ごと（元データそのまま）":
            show = filtered.sort_values(["店舗名", "日時"])
            xcol = "日時"
        elif agg_level == "1時間ごとに合計":
            filtered["hour"] = filtered["日時"].dt.floor("H")
            show = (
                filtered.groupby(["店舗名", "hour"], as_index=False)["デマンド値"]
                .sum()
                .rename(columns={"hour": "日時"})
            )
            xcol = "日時"
        else:
            filtered["日付_d"] = pd.to_datetime(filtered["日付"], errors="coerce").dt.date
            show = (
                filtered.groupby(["店舗名", "日付_d"], as_index=False)["デマンド値"]
                .sum()
                .rename(columns={"日付_d": "日付"})
            )
            xcol = "日付"

        cols_drop = []
        for c in ["日付_only", "時刻_only", "日付_d", "hour"]:
            if c in show.columns and c != xcol:
                cols_drop.append(c)
        if "日付" in show.columns and "日時" in show.columns:
            if xcol == "日時":
                cols_drop.append("日付")
            else:
                cols_drop.append("日時")
        if cols_drop:
            show = show.drop(columns=list(set(cols_drop)))

        tab_graph, tab_sum, tab_table = st.tabs(["📈 グラフ", "📊 合計・集計", "📋 表データ"])

        # --- グラフ ---
        with tab_graph:
            st.subheader("グラフ表示")
            st.caption("えらんだ条件で、デマンド値の推移をグラフで表示します。")
            line = alt.Chart(show).mark_line().encode(
                x=alt.X(f"{xcol}:T", title=xcol),
                y=alt.Y("デマンド値:Q", title="デマンド値"),
                color=alt.Color("店舗名:N", title="店舗"),
                tooltip=list(show.columns),
            ).properties(height=360)
            st.altair_chart(line, use_container_width=True)

        # --- 合計・集計 ---
        with tab_sum:
            st.subheader("合計・集計")
            if not st.session_state.get("show_sum", False):
                st.info("左の「③ 追加ツール」で **「合計・店舗別集計を表示する」** にチェックすると、ここに集計結果が表示されます。")
            else:
                total_sum_value = float(filtered["デマンド値"].sum())
                by_store = (
                    filtered.groupby("店舗名", as_index=False)["デマンド値"]
                    .sum()
                    .sort_values("デマンド値", ascending=False)
                )
                store_count = int(by_store["店舗名"].nunique())
                record_count = int(len(filtered))

                k1, k2, k3 = st.columns(3)
                k1.metric("現在の条件での合計", f"{total_sum_value:,.2f}")
                k2.metric("対象店舗数", f"{store_count:,}")
                k3.metric("データ件数", f"{record_count:,}")

                with st.expander("店舗ごとの合計（ダウンロードできます）", expanded=True):
                    st.dataframe(by_store, use_container_width=True)
                    st.download_button(
                        "店舗ごとの合計をCSVでダウンロード",
                        data=by_store.to_csv(index=False).encode("utf-8-sig"),
                        file_name="店舗別_合計.csv",
                        mime="text/csv",
                    )

                # 任意期間合計
                st.markdown("### ⏱ 任意の期間で合計を出す（時間帯の条件は無視）")

                dt_series_global = pd.to_datetime(data["日時"], errors="coerce")
                valid_dt_global = dt_series_global.dropna()

                if valid_dt_global.empty:
                    st.error("日時の情報が正しく読み込めていません。任意期間の合計は利用できません。")
                    min_dt_global = datetime.combine(min_date, dtime(0, 0))
                    max_dt_global = datetime.combine(max_date, dtime(23, 30))
                else:
                    min_dt_global = valid_dt_global.min()
                    max_dt_global = valid_dt_global.max()

                col_a2, col_b2 = st.columns(2)
                dt_start2 = col_a2.date_input(
                    "開始日", value=min_dt_global.date(), min_value=min_date, max_value=max_date, key="sum_custom_start_date"
                )
                tm_start2 = col_a2.time_input(
                    "開始時刻", value=dtime(0, 0), step=1800, key="sum_custom_start_time"
                )
                dt_end2 = col_b2.date_input(
                    "終了日", value=max_dt_global.date(), min_value=min_date, max_value=max_date, key="sum_custom_end_date"
                )
                tm_end2 = col_b2.time_input(
                    "終了時刻", value=dtime(23, 30), step=1800, key="sum_custom_end_time"
                )

                if st.button("この期間で合計を計算する"):
                    try:
                        start_dt2 = datetime.combine(dt_start2, tm_start2)
                        end_dt2 = datetime.combine(dt_end2, tm_end2)
                        if start_dt2 > end_dt2:
                            st.error("開始日時は終了日時より前にしてください。")
                        else:
                            base2 = data.copy()
                            base2 = base2[base2["店舗名"].astype(str).isin([str(s) for s in selected_stores])]
                            base2["日時"] = pd.to_datetime(base2["日時"], errors="coerce")
                            m2 = (base2["日時"] >= start_dt2) & (base2["日時"] <= end_dt2)
                            base2 = base2.loc[m2]
                            if base2.empty:
                                st.info("指定した任意期間にデータがありません。期間や店舗の条件を少し広げてお試しください。")
                            else:
                                total_custom = float(base2["デマンド値"].sum())
                                c1, c2 = st.columns(2)
                                c1.metric("任意期間の合計（選択した店舗のみ）", f"{total_custom:,.2f}")
                                by_store_custom = (
                                    base2.groupby("店舗名", as_index=False)["デマンド値"]
                                    .sum()
                                    .sort_values("デマンド値", ascending=False)
                                )
                                with st.expander("任意期間：店舗ごとの合計（ダウンロードできます）", expanded=True):
                                    st.dataframe(by_store_custom, use_container_width=True)
                                    st.download_button(
                                        "任意期間_店舗別_合計.csv をダウンロード",
                                        data=by_store_custom.to_csv(index=False).encode("utf-8-sig"),
                                        file_name="任意期間_店舗別_合計.csv",
                                        mime="text/csv",
                                    )
                    except Exception as e:
                        st.error("任意期間の合計計算でエラーが発生しました。")
                        st.exception(e)

        # --- 表データ ---
        with tab_table:
            st.subheader("表データ")
            st.caption("グラフのもとになっているデータを、そのまま表で確認できます。")

            table_df = show.reset_index(drop=True).copy()

            use_slow_row_select = st.checkbox(
                "表の中で行を選んで合計するモード（重くなる場合があります）",
                value=False,
                help="通常はOFFのままをおすすめします。行数が多い場合、このモードをONにすると画面が重くなります。",
            )

            if not use_slow_row_select:
                st.dataframe(table_df, use_container_width=True)
                st.download_button(
                    "表示中のデータをCSVでダウンロード",
                    data=table_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="デマンド値_抽出結果.csv",
                    mime="text/csv",
                )
            else:
                if len(table_df) > 2000:
                    st.warning("行数が多すぎるため、このモードでは画面が非常に重くなる可能性があります。日付・店舗・時間帯をもう少し絞ることをおすすめします。")

                if "デマンド値" not in table_df.columns:
                    st.dataframe(table_df, use_container_width=True)
                    st.info("この集計レベルでは「デマンド値」列が見つかりません。30分／1時間／1日ごと表示でお試しください。")
                else:
                    if "選択" not in table_df.columns:
                        table_df.insert(0, "選択", False)

                    edited_df = st.data_editor(
                        table_df,
                        use_container_width=True,
                        num_rows="fixed",
                        key="data_editor_main",
                    )

                    st.download_button(
                        "表示中のデータをCSVでダウンロード",
                        data=edited_df.drop(columns=["選択"]).to_csv(index=False).encode("utf-8-sig"),
                        file_name="デマンド値_抽出結果.csv",
                        mime="text/csv",
                    )

                    st.markdown("---")
                    st.markdown("### 🧮 チェックした行の合計")

                    if st.button("チェックした行を合計する"):
                        sel_df = edited_df[edited_df["選択"] == True]
                        if not sel_df.empty:
                            sel_sum = float(sel_df["デマンド値"].sum())
                            st.metric("選択した行のデマンド値合計", f"{sel_sum:,.2f}")
                            st.caption(f"選択した行数: {len(sel_df)}")
                        else:
                            st.info("チェックされている行がありません。表の「選択」にチェックを入れてください。")
                    else:
                        st.caption("※ 表の「選択」にチェックを入れてから、「チェックした行を合計する」ボタンを押してください。")

    # ---------- 比較ツール (デマンド 분석) ----------
    if st.session_state.get("show_compare", False):
        st.markdown("---")
        st.subheader("📊 1つの店舗で、期間Aと期間Bを比べる")

        @st.cache_data(show_spinner=True, ttl=1800)
        def load_compare_data(input_file, pq_dir, use_fast):
            if use_fast and parquet_available(pq_dir):
                data_all, _ = load_parquet_all(pq_dir)
                return data_all
            else:
                if input_file is None:
                    return pd.DataFrame(columns=["店舗名", "日付", "時間帯", "デマンド値", "開始時刻", "日時"])
                data_all, _ = load_excel_all(input_file)
                return data_all

        with st.spinner("比較用のデータを準備しています..."):
            cmp_data = load_compare_data(input_file, pq_dir, use_fast)

        if cmp_data.empty:
            st.info("比較に使えるデータがありません。読み込んだファイルをご確認ください。")
        else:
            cmp_data["日時"] = pd.to_datetime(cmp_data["日時"], errors="coerce")
            cmp_data["日付_only_cmp"] = cmp_data["日時"].dt.date

            compare_stores = (
                pd.Series(cmp_data["店舗名"].dropna().astype(str).unique())
                .loc[lambda s: s.ne("") & ~s.str.fullmatch(r"\d+(\.\d+)?")]
                .sort_values()
                .tolist()
            )
            if not compare_stores:
                st.info("店舗名が正しく読み込めていません。データの形式をご確認ください。")
            else:
                csel = st.selectbox(
                    "店舗をえらぶ",
                    compare_stores,
                    index=0,
                    key="compare_store_v11",
                )

                mask_valid_cmp = cmp_data["日付_only_cmp"].notna()
                if not mask_valid_cmp.any():
                    st.error("比較用データの日付が正しく読み込めていません。ファイル形式をご確認ください。")
                    st.stop()

                base_min_date_cmp = cmp_data.loc[mask_valid_cmp, "日付_only_cmp"].min()
                base_max_date_cmp = cmp_data.loc[mask_valid_cmp, "日付_only_cmp"].max()

                time_all = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]

                st.markdown("#### 期間A（基準とする期間）")
                ca1, ca2 = st.columns(2)
                a_start_d = ca1.date_input(
                    "A 開始日",
                    value=base_min_date_cmp,
                    min_value=base_min_date_cmp,
                    max_value=base_max_date_cmp,
                    key="cmp_A_sd_v11",
                )
                a_end_d = ca1.date_input(
                    "A 終了日",
                    value=base_max_date_cmp,
                    min_value=base_min_date_cmp,
                    max_value=base_max_date_cmp,
                    key="cmp_A_ed_v11",
                )
                a_start_t = ca2.selectbox(
                    "A 開始時刻",
                    time_all,
                    index=0,
                    key="cmp_A_st_v11",
                )
                a_end_t = ca2.selectbox(
                    "A 終了時刻",
                    time_all,
                    index=len(time_all) - 1,
                    key="cmp_A_et_v11",
                )

                st.markdown("#### 期間B（比べたい期間）")
                cb1, cb2 = st.columns(2)
                b_start_d = cb1.date_input(
                    "B 開始日",
                    value=base_min_date_cmp,
                    min_value=base_min_date_cmp,
                    max_value=base_max_date_cmp,
                    key="cmp_B_sd_v11",
                )
                b_end_d = cb1.date_input(
                    "B 終了日",
                    value=base_max_date_cmp,
                    min_value=base_min_date_cmp,
                    max_value=base_max_date_cmp,
                    key="cmp_B_ed_v11",
                )
                b_start_t = cb2.selectbox(
                    "B 開始時刻",
                    time_all,
                    index=0,
                    key="cmp_B_st_v11",
                )
                b_end_t = cb2.selectbox(
                    "B 終了時刻",
                    time_all,
                    index=len(time_all) - 1,
                    key="cmp_B_et_v11",
                )

                run_compare = st.button("この2つの期間を比較する")
                if run_compare:
                    try:
                        a_start_dt = pd.Timestamp(f"{a_start_d} {a_start_t}:00")
                        a_end_dt = pd.Timestamp(f"{a_end_d} {a_end_t}:00")
                        b_start_dt = pd.Timestamp(f"{b_start_d} {b_start_t}:00")
                        b_end_dt = pd.Timestamp(f"{b_end_d} {b_end_t}:00")

                        if a_start_dt > a_end_dt or b_start_dt > b_end_dt:
                            st.error("期間A・Bの開始日時は、それぞれ終了日時より前にしてください。")
                        else:
                            base = cmp_data.copy()
                            base = base[base["店舗名"].astype(str) == str(csel)]

                            A = base[(base["日時"] >= a_start_dt) & (base["日時"] <= a_end_dt)].copy()
                            B = base[(base["日時"] >= b_start_dt) & (base["日時"] <= b_end_dt)].copy()

                            if A.empty or B.empty:
                                st.info("期間A または B にデータがありません。日付・時間帯の条件を見直してください。")
                            else:
                                A_sum = float(A["デマンド値"].sum())
                                B_sum = float(B["デマンド値"].sum())
                                diff = B_sum - A_sum
                                pct = (diff / A_sum * 100.0) if A_sum != 0 else np.nan

                                m1, m2, m3, m4 = st.columns(4)
                                m1.metric("A の合計", f"{A_sum:,.2f}")
                                m2.metric("B の合計", f"{B_sum:,.2f}")
                                m3.metric("差（B − A）", f"{diff:,.2f}")
                                m4.metric(
                                    "増減率(%)",
                                    f"{pct:,.2f}" if np.isfinite(pct) else "N/A",
                                )

                                A["日付_only_cmp2"] = A["日時"].dt.date
                                B["日付_only_cmp2"] = B["日時"].dt.date
                                Ag = (
                                    A.groupby("日付_only_cmp2", as_index=False)["デマンド値"]
                                    .sum()
                                    .rename(columns={"デマンド値": "A_合計"})
                                )
                                Bg = (
                                    B.groupby("日付_only_cmp2", as_index=False)["デマンド値"]
                                    .sum()
                                    .rename(columns={"デマンド値": "B_合計"})
                                )

                                Ag = Ag.sort_values("日付_only_cmp2").reset_index(drop=True).assign(相対日=lambda df: df.index + 1)
                                Bg = Bg.sort_values("日付_only_cmp2").reset_index(drop=True).assign(相対日=lambda df: df.index + 1)

                                comp = pd.merge(Ag, Bg, on="相対日", how="outer")
                                comp["A_合計"] = comp["A_合計"].fillna(0.0)
                                comp["B_合計"] = comp["B_合計"].fillna(0.0)
                                comp["差分(B−A)"] = comp["B_合計"] - comp["A_合計"]
                                comp["増減率(%)"] = np.where(
                                    comp["A_合計"] != 0,
                                    (comp["差分(B−A)"] / comp["A_合計"]) * 100.0,
                                    np.nan,
                                )
                                comp = comp.rename(columns={"日付_only_cmp2_x": "A_日付", "日付_only_cmp2_y": "B_日付"})

                                comp["A_日付"] = pd.to_datetime(comp["A_日付"], errors="coerce")
                                comp["B_日付"] = pd.to_datetime(comp["B_日付"], errors="coerce")

                                # A/B 重ねグラフ（相対日）
                                rel_A = comp[["相対日", "A_日付", "A_合計"]].rename(
                                    columns={"A_日付": "日付", "A_合計": "値"}
                                )
                                rel_A["系列"] = "A"
                                rel_B = comp[["相対日", "B_日付", "B_合計"]].rename(
                                    columns={"B_日付": "日付", "B_合計": "値"}
                                )
                                rel_B["系列"] = "B"
                                rel_long = pd.concat([rel_A, rel_B], ignore_index=True)

                                st.markdown("##### 日ごとの合計を、A と B で重ねて比較")
                                line_rel = alt.Chart(rel_long).mark_line().encode(
                                    x=alt.X("相対日:Q", title="相対日（それぞれの期間の1日目を1として並べ替え）"),
                                    y=alt.Y("値:Q", title="日別合計（デマンド値）"),
                                    color=alt.Color("系列:N", title="期間"),
                                    tooltip=["系列", "相対日", "日付:T", "値:Q"],
                                ).properties(height=320)
                                st.altair_chart(line_rel, use_container_width=True)

                                # 実際の日付ベース（参考）
                                real_long = pd.concat(
                                    [
                                        Ag.rename(columns={"A_合計": "値", "日付_only_cmp2": "日付"})
                                        .assign(系列="A")[["日付", "値", "系列"]],
                                        Bg.rename(columns={"B_合計": "値", "日付_only_cmp2": "日付"})
                                        .assign(系列="B")[["日付", "値", "系列"]],
                                    ],
                                    ignore_index=True,
                                )
                                with st.expander("🗓 実際の日付で見たグラフ（参考）"):
                                    line_real = alt.Chart(real_long).mark_line().encode(
                                        x=alt.X("日付:T", title="日付"),
                                        y=alt.Y("値:Q", title="日別合計"),
                                        color=alt.Color("系列:N", title="期間"),
                                        tooltip=["系列", "日付:T", "値:Q"],
                                    ).properties(height=300)
                                    st.altair_chart(line_real, use_container_width=True)

                                with st.expander("🧾 日ごとの比較表（B が A からどれだけ変わったか）", expanded=True):
                                    view = comp[
                                        ["相対日", "A_日付", "B_日付", "A_合計", "B_合計", "差分(B−A)", "増減率(%)"]
                                    ]
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

    # ---------- デバッグ情報 ----------
    with st.expander("🧪 開発・確認用情報（ふだんは見なくてOK）"):
        st.write(
            {
                "use_fast": use_fast,
                "parquet_available": parquet_available(parquet_dir_path(Path.cwd())),
                "auto_source": st.session_state.get("auto_file_source", None),
                "has_partial_cache": bool(st.session_state.get("auto_file_bytes_partial", None)),
                "has_full_cache": bool(st.session_state.get("auto_file_bytes_full", None)),
                "uploaded_file": bool(uploaded_file),
            }
        )

# ======================================================================
#  メニュー３：気温×デマンド分析（相関＋同条件比較＋パターン分析）
# ======================================================================
if menu == "気温×デマンド分析":
    st.sidebar.title("操作パネル（気温×デマンド）")

    st.sidebar.markdown(
        """
**必要な列（日本語の列名）**  

- 日付  
- 時間帯（例：`3:00-3:30`）  
- 気温  
- デマンド値
"""
    )

    temp_file = st.sidebar.file_uploader(
        "気温×デマンド データファイルを選択",
        type=["xlsx", "csv"],
        help="1つのファイルの中に「日付 / 時間帯 / 気温 / デマンド値」が入っている形式を想定しています。",
    )

    st.markdown("## 🌡 気温×デマンド分析")

    if temp_file is None:
        st.info("左のサイドバーから、気温×デマンドのデータファイルをアップロードしてください。")
        st.stop()

    # ---- ファイル読み込み ----
    try:
        if temp_file.name.lower().endswith(".csv"):
            df_td = pd.read_csv(temp_file)
        else:
            df_td = pd.read_excel(temp_file)
    except Exception as e:
        st.error("ファイルの読み込みに失敗しました。形式や文字コードをご確認ください。")
        st.exception(e)
        st.stop()

    # 必要な列チェック
    required_cols = ["日付", "時間帯", "気温", "デマンド値"]
    missing = [c for c in required_cols if c not in df_td.columns]
    if missing:
        st.error(f"次の必要な列が見つかりませんでした: {missing}\n\nファイルの列名を確認してください。")
        st.write("現在の列名一覧:", list(df_td.columns))
        st.stop()

    # 型整形
    df_td = df_td.copy()
    df_td["日付"] = normalize_date_series(df_td["日付"])
    df_td["気温"] = pd.to_numeric(df_td["気温"], errors="coerce")
    df_td["デマンド値"] = pd.to_numeric(df_td["デマンド値"], errors="coerce")

    # 시간帯 문자열 정리
    df_td["時間帯"] = (
        df_td["時間帯"]
        .astype(str)
        .str.replace("\u3000", " ")
        .str.strip()
    )

    # 시각(datetime) 컬럼 만들기
    start_time_str = np.where(
        df_td["時間帯"].str.contains("-"),
        df_td["時間帯"].str.split("-").str[0].str.strip(),
        df_td["時間帯"]
    )
    df_td["開始時刻"] = start_time_str
    df_td["日時"] = pd.to_datetime(
        df_td["日付"].dt.date.astype(str) + " " + df_td["開始時刻"],
        errors="coerce"
    )

    # 시작 시각을 time형으로도 보유 (パターン分析で利用)
    df_td["開始時刻_time"] = pd.to_datetime(
        df_td["開始時刻"], format="%H:%M", errors="coerce"
    ).dt.time

    # 유효 데이터만 사용
    base_mask = df_td["日付"].notna() & df_td["気温"].notna() & df_td["デマンド値"].notna()
    df_td = df_td.loc[base_mask].copy()
    if df_td.empty:
        st.error("有効なデータ（日付・気温・デマンド値）がありません。欠損値を確認してください。")
        st.stop()

    # ------ 季節区分 컬럼 추가 (パターン分析용) ------
    df_td["month"] = df_td["日付"].dt.month
    conds = [
        df_td["month"].isin([11, 12, 1, 2, 3]),
        df_td["month"].isin([4, 5, 6]),
        df_td["month"].isin([7, 8, 9]),
        df_td["month"] == 10,
    ]
    choices = ["冬期", "中間期", "夏期", "10月"]
    df_td["季節区分"] = np.select(conds, choices, default="不明")

    # 탭 3개: 相関 / 同条件 비교 / パターン分析
    tab_corr, tab_cond, tab_pattern = st.tabs(
        ["📈 気温とデマンドの相関を見る", "🔍 同じ条件で2つのパターンを比較", "🧊 パターン分析（季節×時間帯×気温）"]
    )

    # ------------------------------------------------------------------
    # ① 気温とデマンドの相関を見る
    # ------------------------------------------------------------------
    with tab_corr:
        st.subheader("📈 気温とデマンド値の相関を見る（時間帯ごと）")

        st.markdown(
            """
1. 全体としての相関（気温が上がるとデマンドが上がるか／下がるか）  
2. 時間帯ごとの相関（どの時間帯で気温とデマンドが強く連動しているか）  
3. 興味のある時間帯を選んで、散布図で詳しく確認  
"""
        )

        overall_corr = df_td["気温"].corr(df_td["デマンド値"])
        st.metric("全データでの相関係数（気温 vs デマンド値）", f"{overall_corr: .3f}")

        # 시간帯별 상관계수
        corr_by_slot = (
            df_td.groupby("時間帯")
            .apply(lambda g: g["気温"].corr(g["デマンド値"]))
            .dropna()
        )
        corr_df = corr_by_slot.reset_index().rename(columns={0: "相関係数"})

        corr_df["相関の強さ(絶対値)"] = corr_df["相関係数"].abs()
        corr_df = corr_df.sort_values("相関の強さ(絶対値)", ascending=False)

        st.markdown("#### 📊 時間帯ごとの相関（絶対値が大きい順）")
        st.dataframe(corr_df, use_container_width=True)

        st.markdown("#### ⏱ 時間帯別の相関係数（棒グラフ）")
        corr_chart = alt.Chart(corr_df).mark_bar().encode(
            x=alt.X("時間帯:N", sort=None, title="時間帯"),
            y=alt.Y("相関係数:Q", title="相関係数（気温 vs デマンド値）"),
            tooltip=["時間帯", "相関係数", "相関の強さ(絶対値)"],
        ).properties(height=320)
        st.altair_chart(corr_chart, use_container_width=True)

        st.markdown("#### 🔍 興味のある時間帯をえらんで、散布図で確認")
        slots = corr_df["時間帯"].tolist()
        sel_slot = st.selectbox(
            "時間帯をえらぶ",
            options=slots,
            help="例：3:00-3:30 など、同じ時間帯のデータだけを抜き出して表示します。",
        )

        slot_df = df_td[df_td["時間帯"] == sel_slot].copy()
        st.caption(f"選択した時間帯：{sel_slot}（データ件数 {len(slot_df)}）")

        if len(slot_df) < 3:
            st.info("この時間帯のデータ件数が少ないため、相関の傾向がはっきりしない可能性があります。")
        else:
            slot_corr = slot_df["気温"].corr(slot_df["デマンド値"])
            st.metric("この時間帯での相関係数", f"{slot_corr: .3f}")

            scatter = alt.Chart(slot_df).mark_circle(size=60).encode(
                x=alt.X("気温:Q", title="気温"),
                y=alt.Y("デマンド値:Q", title="デマンド値"),
                tooltip=["日付:T", "時間帯", "気温", "デマンド値"],
            )
            reg = scatter.transform_regression("気温", "デマンド値").mark_line()

            st.altair_chart((scatter + reg).properties(height=360), use_container_width=True)

    # ------------------------------------------------------------------
    # ② 同じ条件で2つのパターンを比較
    # ------------------------------------------------------------------
    with tab_cond:
        st.subheader("🔍 同じ条件で2つのパターンを比較")

        st.markdown(
            """
例：  
- 「通常日」と「大雪日」という区分列を用意しておき、  
- 同じ時間帯・同じ気温帯で、デマンド値の平均を比較する  

という使い方を想定しています。
"""
        )

        candidate_cond_cols = [
            c for c in df_td.columns
            if c not in ["日付", "時間帯", "気温", "デマンド値", "日時", "開始時刻", "開始時刻_time", "month", "季節区分"]
        ]

        if not candidate_cond_cols:
            st.info("「通常 / 大雪」などの区分に使えそうな列が見つかりませんでした。ファイルにカテゴリー列（例：天気区分）を追加してからご利用ください。")
        else:
            cond_col = st.selectbox(
                "比較に使う区分の列（例：パターン, 天気区分 など）",
                options=candidate_cond_cols,
                help="この列の中から、2つの値（例：通常 / 大雪）をえらんで比較します。",
            )

            unique_vals = sorted(df_td[cond_col].dropna().astype(str).unique().tolist())
            st.caption(f"列「{cond_col}」のユニーク値: {unique_vals}")

            col_c1, col_c2 = st.columns(2)
            val_A = col_c1.selectbox("パターンA（例：通常）", options=unique_vals)
            val_B = col_c2.selectbox("パターンB（例：大雪）", options=unique_vals, index=min(1, len(unique_vals) - 1))

            st.markdown("#### 比較する条件（同じ条件で絞り込み）")

            all_slots_for_cond = sorted(df_td["時間帯"].dropna().unique().tolist())
            mode_slot = st.radio(
                "時間帯の条件",
                ["全時間帯", "特定の時間帯だけ"],
                horizontal=True,
            )
            if mode_slot == "特定の時間帯だけ":
                cond_slot = st.selectbox("対象とする時間帯", options=all_slots_for_cond)
            else:
                cond_slot = None

            temp_min = float(df_td["気温"].min())
            temp_max = float(df_td["気温"].max())
            t_range = st.slider(
                "気温帯（この範囲に入るデータだけを比較）",
                min_value=float(round(temp_min - 1, 1)),
                max_value=float(round(temp_max + 1, 1)),
                value=(float(round(temp_min, 1)), float(round(temp_max, 1))),
                step=0.5,
            )

            if st.button("この条件でパターンA/Bを比較する"):
                try:
                    base = df_td.copy()
                    base[cond_col] = base[cond_col].astype(str)

                    if cond_slot is not None:
                        base = base[base["時間帯"] == cond_slot]

                    base = base[(base["気温"] >= t_range[0]) & (base["気温"] <= t_range[1])]

                    A_df = base[base[cond_col] == val_A]
                    B_df = base[base[cond_col] == val_B]

                    if A_df.empty or B_df.empty:
                        st.warning("指定した条件のもとで、A または B のデータがありません。時間帯・気温帯・区分を見直してください。")
                    else:
                        def summarize(name, df):
                            return {
                                "パターン": name,
                                "件数": int(len(df)),
                                "平均デマンド": float(df["デマンド値"].mean()),
                                "平均気温": float(df["気温"].mean()),
                            }

                        sum_A = summarize(val_A, A_df)
                        sum_B = summarize(val_B, B_df)
                        sum_df = pd.DataFrame([sum_A, sum_B])

                        st.markdown("#### 結果サマリー（同じ条件下での比較）")
                        st.dataframe(sum_df, use_container_width=True)

                        diff_demand = sum_B["平均デマンド"] - sum_A["平均デマンド"]
                        diff_temp = sum_B["平均気温"] - sum_A["平均気温"]

                        k1, k2 = st.columns(2)
                        k1.metric("B − A の平均デマンド差", f"{diff_demand: .3f}")
                        k2.metric("B − A の平均気温差", f"{diff_temp: .3f}")

                        st.markdown("#### 平均デマンド値の比較（棒グラフ）")
                        bar = alt.Chart(sum_df).mark_bar().encode(
                            x=alt.X("パターン:N", title="パターン"),
                            y=alt.Y("平均デマンド:Q", title="平均デマンド値"),
                            tooltip=["パターン", "件数", "平均デマンド", "平均気温"],
                        ).properties(height=320)
                        st.altair_chart(bar, use_container_width=True)

                        with st.expander("散布図で詳しく見る（気温 vs デマンド値）"):
                            A_df["パターン"] = val_A
                            B_df["パターン"] = val_B
                            scatter_df = pd.concat([A_df, B_df], ignore_index=True)

                            scat = alt.Chart(scatter_df).mark_circle(size=60).encode(
                                x=alt.X("気温:Q", title="気温"),
                                y=alt.Y("デマンド値:Q", title="デマンド値"),
                                color=alt.Color("パターン:N", title="パターン"),
                                tooltip=["日付:T", "時間帯", "パターン", "気温", "デマンド値"],
                            ).properties(height=360)
                            st.altair_chart(scat, use_container_width=True)

                        st.download_button(
                            "この比較結果をCSVでダウンロード",
                            data=sum_df.to_csv(index=False).encode("utf-8-sig"),
                            file_name=f"気温デマンド_同条件比較_{val_A}_vs_{val_B}.csv",
                            mime="text/csv",
                        )

                except Exception as e:
                    st.error("同条件比較の処理でエラーが発生しました。")
                    st.exception(e)

    # ------------------------------------------------------------------
    # ③ パターン分析（冬期・夏期・中間期・10月 × 時間帯 × 気温帯）
    # ------------------------------------------------------------------
    with tab_pattern:
        st.subheader("🧊 パターン分析（季節×時間帯×気温＋外れ値除外）")

        st.markdown(
            """
**パターン分析の考え方**

1. 1年間を 4つの季節に分ける  
   - 冬期：11〜3月  
   - 中間期：4〜6月  
   - 夏期：7〜9月  
   - 10月：10月だけ別  

2. 同じ季節・同じ時間帯・同じ気温帯のデータを1つの「パターン」とみなす  

3. そのパターンの中で：  
   - まず平均を出す  
   - その平均から ±15% 以上離れているデマンド値を外れ値として除外  
   - 残ったデータでもう一度平均を出す（これを最終平均とする）
"""
        )

        # 季節 선택
        all_seasons = ["冬期", "中間期", "夏期", "10月"]
        sel_seasons = st.multiselect(
            "対象とする季節区分",
            options=all_seasons,
            default=all_seasons,
            help="分析したい季節だけを選ぶこともできます。",
        )
        if not sel_seasons:
            st.info("少なくとも1つの季節区分を選んでください。")
            st.stop()

        # 시간帯 범위 선택
        st.markdown("#### 時間帯の条件")
        time_mode_p = st.radio(
            "時間帯のえらび方",
            ["全時間帯", "特定の時間帯範囲（開始〜終了）"],
            horizontal=True,
        )

        unique_times = sorted(
            [t for t in df_td["開始時刻_time"].dropna().unique().tolist()],
            key=lambda x: (x.hour, x.minute)
        )

        if time_mode_p == "特定の時間帯範囲（開始〜終了）" and unique_times:
            col_tp1, col_tp2 = st.columns(2)
            t_start_sel = col_tp1.selectbox(
                "開始時刻",
                options=unique_times,
                format_func=lambda t: t.strftime("%H:%M"),
            )
            t_end_sel = col_tp2.selectbox(
                "終了時刻",
                options=unique_times,
                index=len(unique_times) - 1,
                format_func=lambda t: t.strftime("%H:%M"),
            )
            if t_start_sel > t_end_sel:
                st.error("開始時刻は終了時刻より前にしてください。")
                st.stop()
        else:
            t_start_sel = None
            t_end_sel = None

        # 気温帯 선택
        st.markdown("#### 気温帯の条件")
        temp_min2 = float(df_td["気温"].min())
        temp_max2 = float(df_td["気温"].max())
        t_range2 = st.slider(
            "気温帯（この範囲に入るデータだけをパターンに含める）",
            min_value=float(round(temp_min2 - 1, 1)),
            max_value=float(round(temp_max2 + 1, 1)),
            value=(float(round(temp_min2, 1)), float(round(temp_max2, 1))),
            step=0.5,
        )

        if st.button("この条件でパターン分析を実行する"):
            try:
                base_p = df_td.copy()
                base_p = base_p[base_p["季節区分"].isin(sel_seasons)]

                if t_start_sel is not None and t_end_sel is not None:
                    base_p = base_p[
                        (base_p["開始時刻_time"] >= t_start_sel)
                        & (base_p["開始時刻_time"] <= t_end_sel)
                    ]

                base_p = base_p[
                    (base_p["気温"] >= t_range2[0])
                    & (base_p["気温"] <= t_range2[1])
                ]

                if base_p.empty:
                    st.warning("指定した季節・時間帯・気温帯に該当するデータがありません。条件を見直してください。")
                else:
                    # 1차 평균
                    mean1 = float(base_p["デマンド値"].mean())
                    # 15% 이상 차이 나는 값 제거
                    lower = mean1 * 0.85
                    upper = mean1 * 1.15
                    mask_keep = (base_p["デマンド値"] >= lower) & (base_p["デマンド値"] <= upper)
                    kept = base_p[mask_keep]
                    dropped = base_p[~mask_keep]

                    n_total = len(base_p)
                    n_kept = len(kept)
                    n_drop = len(dropped)
                    mean2 = float(kept["デマンド値"].mean()) if n_kept > 0 else np.nan
                    drop_ratio = n_drop / n_total * 100.0 if n_total > 0 else 0.0

                    st.markdown("#### 集計結果（この条件でのパターン）")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("元データ件数", f"{n_total:,}")
                    c2.metric("外れ値除外後の件数", f"{n_kept:,}")
                    c3.metric("一次平均（除外前）", f"{mean1: .3f}")
                    c4.metric("最終平均（除外後）", f"{mean2: .3f}")

                    st.caption(f"外れ値として除外された割合: {drop_ratio: .1f} %")

                    with st.expander("残ったデータ（外れ値除外後）を確認する", expanded=False):
                        st.dataframe(
                            kept[["日付", "時間帯", "気温", "デマンド値", "季節区分"]],
                            use_container_width=True,
                        )
                        st.download_button(
                            "外れ値除外後のデータをCSVでダウンロード",
                            data=kept.to_csv(index=False).encode("utf-8-sig"),
                            file_name="パターン分析_残データ.csv",
                            mime="text/csv",
                        )

                    with st.expander("外れ値として除外されたデータを確認する", expanded=False):
                        if dropped.empty:
                            st.write("外れ値として除外されたデータはありません。")
                        else:
                            st.dataframe(
                                dropped[["日付", "時間帯", "気温", "デマンド値", "季節区分"]],
                                use_container_width=True,
                            )
                            st.download_button(
                                "外れ値として除外されたデータをCSVでダウンロード",
                                data=dropped.to_csv(index=False).encode("utf-8-sig"),
                                file_name="パターン分析_外れ値.csv",
                                mime="text/csv",
                            )

                    # 간단한 히스토그램 (デマンド値 분포)
                    with st.expander("デマンド値の分布（外れ値除外後）", expanded=False):
                        hist = alt.Chart(kept).mark_bar().encode(
                            x=alt.X("デマンド値:Q", bin=alt.Bin(maxbins=30), title="デマンド値"),
                            y=alt.Y("count():Q", title="件数"),
                            tooltip=["count()"],
                        ).properties(height=300)
                        st.altair_chart(hist, use_container_width=True)

            except Exception as e:
                st.error("パターン分析の処理でエラーが発生しました。")
                st.exception(e)
