"""DataFrame -> 带高亮的 Excel bytes（PRD §6.5）。

要求：
- 导出【始终为全量 N 行】，不受页面渲染行数限制
- 异常行按标签填充底色
- 附第二个 Sheet「行业标准」（原文 + 来源 + 置信度）
- >1000 行自动切换 XlsxWriter（快 3~5 倍），用 conditional_format 而非逐格填色
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from config import (
    HIGHLIGHT_RED, HIGHLIGHT_GOLD, HIGHLIGHT_GREEN, EXCEL_ENGINE_SWITCH_ROWS,
    COL_TAG, COL_TAG_PLAIN, PRODUCT_TAGS, AD_TAGS,
)
from utils.diagnose import strip_tag_plain

_SHEET_MAIN = "诊断结果"
_SHEET_STD = "行业标准"


def _col_letter(i: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA。"""
    letters = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        letters = chr(65 + r) + letters
    return letters


def _standard_sheet_text(std_text: str) -> pd.DataFrame:
    lines = (std_text or "").split("\n")
    return pd.DataFrame({"行业标准（AI 联网自主调研生成，仅供参考）": lines})


def to_excel_bytes(
    df: pd.DataFrame,
    std_text: str = "",
    red_tags: list[str] | None = None,
    gold_tags: list[str] | None = None,
    sheet_main: str = _SHEET_MAIN,
) -> bytes:
    """导出全量 DataFrame 为带高亮的 xlsx bytes。"""
    buf = io.BytesIO()
    n = len(df)
    engine = "xlsxwriter" if n > EXCEL_ENGINE_SWITCH_ROWS else "openpyxl"

    out = df.copy()
    if COL_TAG in out.columns and COL_TAG_PLAIN not in out.columns:
        # 插在「诊断标签」之后，而不是之前，保持列顺序可读
        out.insert(out.columns.get_loc(COL_TAG) + 1, COL_TAG_PLAIN,
                   out[COL_TAG].map(strip_tag_plain))

    # 供 xlsxwriter 读取的下标（openpyxl 分支不需要）
    if isinstance(out.index, pd.RangeIndex) or np.issubdtype(out.index.dtype, np.integer):
        out = out.reset_index(drop=True)

    try:
        with pd.ExcelWriter(buf, engine=engine) as w:
            out.to_excel(w, index=False, sheet_name=sheet_main)
            _standard_sheet_text(std_text).to_excel(w, index=False, sheet_name=_SHEET_STD)

            if engine == "xlsxwriter":
                _paint_xlsxwriter(w, out, sheet_main, red_tags, gold_tags)
            else:
                pass  # openpyxl 分支在文件写完后统一处理（需要 workbook 对象）

        if engine == "openpyxl":
            buf = _paint_openpyxl(buf, out, sheet_main, red_tags, gold_tags)

    except ImportError:
        # 引擎缺失时降级：不带样式，但保证导出可用
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            out.to_excel(w, index=False, sheet_name=sheet_main)
            _standard_sheet_text(std_text).to_excel(w, index=False, sheet_name=_SHEET_STD)

    return buf.getvalue()


def _tag_patterns(red_tags, gold_tags):
    red = list(red_tags or []) or [t for t in PRODUCT_TAGS if t.startswith("🔴") or t.startswith("🟠")] \
        + [t for t in AD_TAGS if t.startswith("🔴") or t.startswith("⚪")]
    gold = list(gold_tags or []) or [t for t in AD_TAGS if t.startswith("🟡")]
    return [strip_tag_plain(t) for t in red if strip_tag_plain(t)], \
           [strip_tag_plain(t) for t in gold if strip_tag_plain(t)]


def _paint_xlsxwriter(writer, df: pd.DataFrame, sheet_main: str, red_tags, gold_tags) -> None:
    """XlsxWriter：条件格式（按"标签文本"纯文本列匹配，规避 emoji 代理对问题）。"""
    if COL_TAG_PLAIN not in df.columns:
        return
    wb = writer.book
    ws = writer.sheets[sheet_main]
    idx = df.columns.get_loc(COL_TAG_PLAIN)
    letter = _col_letter(idx)
    last_row = max(len(df), 1)
    last_col = max(len(df.columns) - 1, 0)
    rng = (1, 0, last_row, last_col)

    reds, golds = _tag_patterns(red_tags, gold_tags)
    fmt_red = wb.add_format({"bg_color": HIGHLIGHT_RED})
    fmt_gold = wb.add_format({"bg_color": HIGHLIGHT_GOLD})
    fmt_green = wb.add_format({"bg_color": HIGHLIGHT_GREEN})

    def _cf(txt, fmt):
        ws.conditional_format(*rng, {
            "type": "formula",
            # 「标签文本」列为纯文本（已去 emoji），规避 Excel 中 emoji 代理对匹配失败。
            # 注意 criteria 里的行号是相对数据区首行的偏移：rng 从第 1 行（0-based=1）开始，
            # 表头占 row 0，因此首行数据在 Excel 第 2 行。这里必须用「绝对定位到该行」的写法。
            "criteria": f'=EXACT(${letter}2,"{txt}")',
            "format": fmt,
        })

    for t in reds:
        _cf(t, fmt_red)
    for t in golds:
        _cf(t, fmt_gold)
    _cf("表现良好", fmt_green)

    # 列宽：名称类列放宽，其余自适应
    ws.set_column(0, min(len(df.columns) - 1, 40), 16)


def _paint_openpyxl(buf: io.BytesIO, df: pd.DataFrame, sheet_main: str,
                    red_tags, gold_tags) -> io.BytesIO:
    """openpyxl：小表逐行填色。返回新的 BytesIO。"""
    from openpyxl import load_workbook
    from openpyxl.styles import PatternFill

    buf.seek(0)
    wb = load_workbook(buf)
    ws = wb[sheet_main]

    reds, golds = _tag_patterns(red_tags, gold_tags)
    fill_red = PatternFill("solid", start_color="FF" + HIGHLIGHT_RED.lstrip("#"))
    fill_gold = PatternFill("solid", start_color="FF" + HIGHLIGHT_GOLD.lstrip("#"))

    if COL_TAG_PLAIN not in df.columns:
        return buf

    tag_col = list(df.columns).index(COL_TAG_PLAIN) + 1
    for r in range(2, len(df) + 2):
        val = ws.cell(row=r, column=tag_col).value
        val = strip_tag_plain(val) if val else ""
        fill = None
        if val in reds:
            fill = fill_red
        elif val in golds:
            fill = fill_gold
        if fill:
            for c in range(1, len(df.columns) + 1):
                ws.cell(row=r, column=c).fill = fill

    new_buf = io.BytesIO()
    wb.save(new_buf)
    new_buf.seek(0)
    return new_buf


def build_display_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """按指定列顺序组装展示/导出用的 DataFrame（缺失列自动跳过）。"""
    existing = [c for c in cols if c in df.columns]
    return df[existing].copy()


def report_filename(prefix: str) -> str:
    from datetime import datetime
    return f"{prefix}_{datetime.now().strftime('%Y%m%d')}.xlsx"
