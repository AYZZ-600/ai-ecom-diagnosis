"""数据清洗（对应 PRD §3.2 / §4.2）。

提供两套：
- 标量版：pct_to_float / money_to_float / int_col（PRD 原样实现）
- 向量版：pct_series / money_series / int_series（PRD §6.6 要求禁止 iterrows，全部向量化）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import NA_TOKENS, MAX_NAME_LEN, MAX_SUGGESTION_LEN

_NA_SET = set(NA_TOKENS)


# ==========================================================
# 标量版（PRD §3.2 原样固化）
# ==========================================================

def pct_to_float(x):
    """'3.5%' -> 0.035；'-'、''、NaN -> 0.0"""
    if pd.isna(x):
        return 0.0
    s = str(x).strip().replace("%", "").replace(",", "")
    if s in _NA_SET:
        return 0.0
    try:
        return float(s) / 100.0
    except ValueError:
        return 0.0


def int_col(x):
    """'-' -> 0; '828' -> 828"""
    if pd.isna(x):
        return 0
    s = str(x).strip().replace(",", "")
    if s in _NA_SET:
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def money_to_float(x):
    """'￥296.08' -> 296.08；'-'/''/NaN -> 0.0"""
    if pd.isna(x):
        return 0.0
    s = str(x).strip()
    for ch in ("￥", "¥", "$", "£", "€", ","):
        s = s.replace(ch, "")
    if s in _NA_SET:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


# ==========================================================
# 向量版（5000 行规模下必须向量化，禁止逐行 apply）
# ==========================================================

def _clean_numeric_string(s: pd.Series, drops: str) -> pd.Series:
    """统一的字符串清理 + 缺失值归零，返回 float Series。"""
    out = s.astype("object").where(s.notna(), None)
    out = out.astype(str).str.strip()
    for ch in drops:
        out = out.str.replace(ch, "", regex=False)
    out = out.replace(list(_NA_SET) + ["None"], np.nan)
    return pd.to_numeric(out, errors="coerce").fillna(0.0)


def pct_series(s: pd.Series) -> pd.Series:
    """百分比字符串列 -> float 小数。'3.5%' -> 0.035。

    若原列已是数值（如 3.5 表示百分数），自动按百分数处理；
    若为 0.035 这种小数形式（原值 <= 1 且无 % 号），保持原样。
    """
    if pd.api.types.is_numeric_dtype(s):
        return _maybe_fraction(s)
    return _clean_numeric_string(s, "%,，") / 100.0


def _maybe_fraction(s: pd.Series) -> pd.Series:
    """数值型百分比列：区分 3.5（百分制）与 0.035（小数制）。

    仅当整列最大值 <= 1 时才认为是小数制，否则视为百分制并除以 100。
    """
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    if len(s) and s.max() <= 1.0:
        return s
    return s / 100.0


def money_series(s: pd.Series) -> pd.Series:
    """金额列 -> float。支持 ￥ / ¥ / $ / 千分位逗号。"""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0.0)
    return _clean_numeric_string(s, "￥¥$£€,，" )


def int_series(s: pd.Series) -> pd.Series:
    """整数列 -> int64。"""
    v = _clean_numeric_string(s, "%,，")
    return v.round().astype("int64")


def float_series(s: pd.Series) -> pd.Series:
    """普通数值列 -> float。"""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0.0)
    return _clean_numeric_string(s, ",，")


def str_series(s: pd.Series) -> pd.Series:
    """ID 等字符串列：转 str 并去首尾空格，避免科学计数法。"""
    return s.astype("object").where(s.notna(), "").astype(str).str.strip()


# ==========================================================
# 文本工具
# ==========================================================

def truncate(s, n: int) -> str:
    """按字符数截断到【最终不超过 n 个字符】（含省略号）。

    测试输入 "…负 <15 字单元测试…" 超长时必须是 n-1 个字 + "…"，
    否则总长会变成 n+1，违反 PRD「15 个字以内」的硬性要求。
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    t = str(s)
    t = t.replace("\n", " ").strip()
    if len(t) <= n:
        return t
    return t[: max(n - 1, 0)] + "…"


def truncate_series(s: pd.Series, n: int) -> pd.Series:
    """向量化截断（同样含省略号后总长 <= n）。"""
    t = s.astype("object").where(s.notna(), "").astype(str)
    t = t.str.replace("\n", " ", regex=False).str.strip()
    return np.where(t.str.len() > n, t.str[: max(n - 1, 0)] + "…", t)


def clip_suggestion(s) -> str:
    """优化建议按字符数截断到 MAX_SUGGESTION_LEN。"""
    return truncate(s, MAX_SUGGESTION_LEN)


def safe_id(x) -> str:
    """产品ID 统一转字符串，防止科学计数法。"""
    if pd.isna(x) if not isinstance(x, (list, tuple)) else False:
        return ""
    return str(x).strip()


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """安全除法：分母为 0 或 NaN -> NaN（表示"样本不足"，不参与判定）。"""
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    denom = b.replace(0, np.nan)
    return a / denom


def fmt_pct(v) -> str:
    """0.035 -> '3.50%'；NaN -> '样本不足'。"""
    if v is None or (isinstance(v, float) and (pd.isna(v) or np.isnan(v))):
        return "样本不足"
    return f"{v * 100:.2f}%"


def fmt_money(v) -> str:
    """296.08 -> '￥296.08'；NaN -> '-'。"""
    if v is None or (isinstance(v, float) and (pd.isna(v) or np.isnan(v))):
        return "-"
    return f"￥{v:,.2f}"


def json_safe(obj):
    """递归清理 NaN / inf / NaT / numpy 标量，产出 json.dumps 合法的对象。

    必要性：pandas float 列中 None 会被打回 NaN，
    直接 to_dict 会得到 {"k": nan}，json.dumps 输出非法 JSON 的 `NaN` 字面量，
    导致千问解析失败。这里统一转成 None（-> JSON null）。
    """
    import numpy as _np
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if obj is None:
        return None
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else round(v, 6)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    try:
        if pd.isna(obj):        # NaT / NA
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, bool)):
        return obj
    return str(obj)


def fmt_int(v) -> str:
    if v is None or (isinstance(v, float) and (pd.isna(v) or np.isnan(v))):
        return "-"
    return f"{int(v):,}"
