"""Excel 读取 / 校验 / 清洗 / 派生指标 / 汇总统计（两个 Tab 共用）。

对应 PRD §3.1~§3.3、§4.1~§4.3、§6.6。
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from config import (
    SHEET_PRODUCT, DEFAULT_HEADER_ROW_PRODUCT, COLS_PRODUCT_REQUIRED, COLS_PRODUCT_OPTIONAL,
    SHEET_AD, DEFAULT_HEADER_ROW_AD, COLS_AD_REQUIRED, COLS_AD_OPTIONAL, CATEGORY_LEXICON,
)
from utils.clean import (
    float_series, int_series, money_series, pct_series, str_series, safe_div,
    truncate_series, json_safe,
)

MAX_NAME_LEN = 60

# 词典按长度降序，保证 "shower chair" 优先于 "chair" 这类短词命中
_LEX_SORTED = sorted(CATEGORY_LEXICON, key=len, reverse=True)


def extract_core_category(name) -> str:
    """从商品标题中抽取【核心品类词】（如 Shower Chair / Grab Bar / Baseboard）。

    用于约束 AI 不得改变产品的基础定义：建议必须沿用这个词。
    抽不到时返回空字符串（表示不做约束）。
    """
    t = str(name or "").lower()
    for term in _LEX_SORTED:
        if term in t:
            return term
    return ""


def core_category_series(names: pd.Series) -> pd.Series:
    """向量化抽取核心品类词。"""
    t = names.astype("object").where(names.notna(), "").astype(str).str.lower()
    out = pd.Series([""] * len(t), index=t.index, dtype="object")
    for term in _LEX_SORTED:
        hit = t.str.contains(term, regex=False) & (out == "")
        if hit.any():
            out = out.mask(hit, term)
    return out


# ==========================================================
# 读取
# ==========================================================

def get_sheet_names(file) -> list[str]:
    """列出文件内所有 Sheet 名（用于 Sheet 名不匹配时给用户选择）。"""
    try:
        _seek(file)
        xl = pd.ExcelFile(file)
        return list(xl.sheet_names)
    except Exception:
        return []


def _seek(file):
    """UploadedFile 可能被读过，重置指针。"""
    try:
        file.seek(0)
    except Exception:
        pass


def read_raw(file, sheet_name: str, header_row: int) -> pd.DataFrame:
    """按指定 Sheet 与表头行号读取原始 DataFrame。

    PRD §0.1 纠偏：表头在第 header_row 行 -> skiprows = header_row - 1
    """
    _seek(file)
    df = pd.read_excel(
        file,
        sheet_name=sheet_name,
        skiprows=max(int(header_row) - 1, 0),
        header=0,
        dtype={"产品ID": str, "product_id": str},
    )
    # 去除列名首尾空格（实测存在"分享人数 "）
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")          # 删除全空行
    df.columns = [str(c).strip() for c in df.columns]   # 冗余一次保险
    return df


def check_required(df: pd.DataFrame, required: list[str]) -> list[str]:
    """返回缺失的必需列。"""
    return [c for c in required if c not in df.columns]


def pick_missing_sheet(file, want: str) -> tuple[str, bool, list[str]]:
    """Sheet 不存在 -> 取第 1 个 Sheet，并返回 (实际用名, 是否发生回退, 全部 sheet 列表)。"""
    sheets = get_sheet_names(file)
    if not sheets:
        return want, False, []
    if want in sheets:
        return want, False, sheets
    return sheets[0], True, sheets


# ==========================================================
# 模块一：商品
# ==========================================================

def prepare_product(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """清洗商品表 + 计算派生指标。返回 (df, meta)。

    meta 记录被丢弃的行数、数据质量分布等，用于页面提示。
    """
    meta = {"rows_raw": len(raw)}
    if "产品ID" not in raw.columns:
        return raw, meta

    cols_exist = [c for c in (COLS_PRODUCT_REQUIRED + COLS_PRODUCT_OPTIONAL) if c in raw.columns]
    df = raw[cols_exist].copy()

    # --- 产品ID ---
    df["产品ID"] = str_series(df["产品ID"])
    before = len(df)
    df = df[df["产品ID"].notna() & (df["产品ID"] != "") & (df["产品ID"] != "nan")].copy()
    meta["rows_dropped_no_id"] = before - len(df)

    # --- 去重（保留首次出现）---
    before = len(df)
    df = df.drop_duplicates(subset=["产品ID"], keep="first").copy()
    meta["rows_dropped_dup"] = before - len(df)

    # --- 数值列 ---
    for c in ["搜索曝光次数", "搜索点击次数", "询盘个数", "访问人数", "询盘人数",
              "TM咨询人数", "提交订单个数", "收藏人数"]:
        if c in df.columns:
            df[c] = int_series(df[c])

    for c in ["搜索点击率", "询盘率"]:
        if c in df.columns:
            df[c] = pct_series(df[c])

    if "近90天商机人数" in df.columns:
        df["近90天商机人数"] = int_series(df["近90天商机人数"])

    # --- 派生指标（PRD §3.3，全部向量化）---
    df["点击率_计算"] = safe_div(df["搜索点击次数"], df["搜索曝光次数"])
    df["询盘率_计算"] = safe_div(df["询盘个数"], df["搜索点击次数"])

    # ★ 商机口径（用户确认）：买家多数走 TM 而非询盘，
    #   故核心转化判据用「商机数 = 询盘个数 + TM咨询人数」。
    #   注意：平台导出的「近90天商机人数」实测全为 "-"，不可用。
    if "TM咨询人数" not in df.columns:
        df["TM咨询人数"] = 0
    df["商机数"] = df["询盘个数"].astype(int) + df["TM咨询人数"].astype(int)
    df["商机率_计算"] = safe_div(df["商机数"], df["搜索点击次数"])
    df["曝光分位"] = df["搜索曝光次数"].rank(pct=True)
    # 核心品类词：约束 AI 不得改变产品的基础定义
    df["核心品类词"] = core_category_series(df["产品名称"])

    # 平台原生点击率缺失时用计算值补齐（口径统一为 点击/曝光）
    if "搜索点击率" not in df.columns:
        df["搜索点击率"] = df["点击率_计算"]
    if "询盘率" not in df.columns:
        df["询盘率"] = df["询盘率_计算"]

    # --- 数据质量标记 ---
    quality = pd.Series([""] * len(df), index=df.index, dtype="object")
    quality = quality.mask(df["搜索曝光次数"] == 0, "曝光为0")
    quality = quality.mask((df["搜索曝光次数"] > 0) & (df["搜索点击次数"] == 0), "点击为0")
    df["_数据质量"] = quality

    meta["rows"] = len(df)
    meta["zero_exposure"] = int((df["搜索曝光次数"] == 0).sum())
    meta["zero_click"] = int((df["搜索点击次数"] == 0).sum())
    df = df.reset_index(drop=True)
    return df, meta


def summarize_product(df: pd.DataFrame) -> dict:
    """全店汇总指标卡数据（PRD §3.3）。"""
    imp = int(df["搜索曝光次数"].sum())
    clk = int(df["搜索点击次数"].sum())
    inq = int(df["询盘个数"].sum())
    opp = int(df["商机数"].sum()) if "商机数" in df.columns else inq
    n = len(df)
    return {
        "n_total": n,
        "impressions": imp,
        "clicks": clk,
        "inquiries": inq,
        # ★ 核心转化指标改为商机
        "opportunities": opp,
        "visitors": int(df["访问人数"].sum()) if "访问人数" in df.columns else 0,
        "opp_all": opp / clk if clk else float("nan"),
        "ctr_all": clk / imp if imp else float("nan"),
        "inq_all": inq / clk if clk else float("nan"),
        "exp_p50": float(df["搜索曝光次数"].median()),
        "clk_p50": float(df["搜索点击次数"].median()),
        "exp_p75": float(df["搜索曝光次数"].quantile(0.75)),
        "clk_p75": float(df["搜索点击次数"].quantile(0.75)),
        "zero_inq_ratio": float((df["询盘个数"] == 0).mean()) if n else 0.0,
        "zero_opp_ratio": float((df["商机数"] == 0).mean()) if n and "商机数" in df.columns else 0.0,
        "zero_opp_click_ratio": (float(df.loc[df["商机数"] == 0, "搜索点击次数"].sum() / clk)
                                 if clk and "商机数" in df.columns else 0.0),
    }


def product_rows_for_llm(df: pd.DataFrame) -> list[dict]:
    """组装送给 AI 的精简字段（PRD §3.5.2：产品名截断 60 字符）。"""
    sub = pd.DataFrame({
        "product_id": df["产品ID"].astype(str),
        "name": truncate_series(df["产品名称"], MAX_NAME_LEN),
        "impressions": df["搜索曝光次数"].astype(int),
        "clicks": df["搜索点击次数"].astype(int),
        "ctr": df["点击率_计算"].round(4),
        # ★ 商机口径：opportunities = 询盘个数 + TM咨询人数
        "opportunities": (df["商机数"].astype(int) if "商机数" in df.columns
                          else df["询盘个数"].astype(int)),
        "opportunity_rate": (df["商机率_计算"].round(4) if "商机率_计算" in df.columns
                             else df["询盘率_计算"].round(4)),
    })
    if "访问人数" in df.columns:
        sub["visitors"] = df["访问人数"].astype(int)
    if "核心品类词" in df.columns:
        sub["core_category"] = df["核心品类词"]
    if "_local_tag" in df.columns:
        sub["local_tag"] = df["_local_tag"]
    return json_safe(sub.astype(object).where(pd.notna(sub), None).to_dict(orient="records"))


# ==========================================================
# 模块二：广告
# ==========================================================

def prepare_ad(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """清洗广告表 + 计算派生指标。返回 (df, meta)。"""
    meta = {"rows_raw": len(raw)}
    if "产品ID" not in raw.columns:
        return raw, meta

    cols_exist = [c for c in (COLS_AD_REQUIRED + COLS_AD_OPTIONAL) if c in raw.columns]
    df = raw[cols_exist].copy()

    df["产品ID"] = str_series(df["产品ID"])
    before = len(df)
    df = df[df["产品ID"].notna() & (df["产品ID"] != "") & (df["产品ID"] != "nan")].copy()
    meta["rows_dropped_no_id"] = before - len(df)
    before = len(df)
    df = df.drop_duplicates(subset=["产品ID"], keep="first").copy()
    meta["rows_dropped_dup"] = before - len(df)

    # --- 金额列（含 ￥ 前缀）---
    for c in ["花费", "商机成本", "点击成本"]:
        if c in df.columns:
            df[c] = money_series(df[c])

    # --- 整数列 ---
    for c in ["商机量", "点击量", "曝光量", "L1+商机量", "L1+点击量", "询盘量", "订单量", "TM咨询量"]:
        if c in df.columns:
            df[c] = int_series(df[c])

    # --- 百分比列 ---
    for c in ["点击率", "商机转化率", "L1+买家点击占比"]:
        if c in df.columns:
            df[c] = pct_series(df[c])

    # ★ 商机口径对齐（用户确认）：商机数 = 商机量 + TM咨询量
    #   与商品模块保持一致，都把 TM 路径计入转化。
    if "TM咨询量" not in df.columns:
        df["TM咨询量"] = 0
    df["商机数"] = df["商机量"].astype(int) + df["TM咨询量"].astype(int)

    # --- 派生指标（PRD §4.3，分子统一用商机数）---
    df["商机转化率_计算"] = safe_div(df["商机数"], df["点击量"])
    df["CPC_计算"] = safe_div(df["花费"], df["点击量"])
    df["CTR_计算"] = safe_div(df["点击量"], df["曝光量"])
    df["单次商机成本_计算"] = safe_div(df["花费"], df["商机数"])
    df["是否投放"] = df["花费"] > 0
    df["核心品类词"] = core_category_series(df["商品信息"])

    # 平台"点击率"列缺失时补计算值
    if "点击率" not in df.columns:
        df["点击率"] = df["CTR_计算"]
    if "商机转化率" not in df.columns:
        df["商机转化率"] = df["商机转化率_计算"]
    if "商机成本" not in df.columns:
        df["商机成本"] = df["单次商机成本_计算"]

    meta["rows"] = len(df)
    meta["rows_invested"] = int(df["是否投放"].sum())
    df = df.reset_index(drop=True)
    return df, meta


def summarize_ad(df: pd.DataFrame) -> dict:
    """广告汇总指标卡（PRD §4.3）。"""
    invested = df[df["是否投放"]]
    total_cost = float(df["花费"].sum())
    opp_col = "商机数" if "商机数" in df.columns else "商机量"
    zero_lead_cost = float(invested.loc[invested[opp_col] == 0, "花费"].sum())
    n_clicks = int(df["点击量"].sum())
    n_opp = int(df["商机数"].sum()) if "商机数" in df.columns else int(df["商机量"].sum())
    return {
        "n": len(df),
        "n_invested": int(invested.shape[0]),
        "total_cost": total_cost,
        "total_leads": n_opp,
        "total_clicks": n_clicks,
        "total_impressions": int(df["曝光量"].sum()),
        "cpc_all": total_cost / n_clicks if n_clicks else float("nan"),
        "lead_rate_all": n_opp / n_clicks if n_clicks else float("nan"),
        "zero_opp_click_ratio": (float(df.loc[df["商机数"] == 0, "点击量"].sum() / n_clicks)
                                 if n_clicks and "商机数" in df.columns else 0.0),
        # 分位数只在「实际投放」的行内计算：花费=0 的行恒为 🟢 正常，不应抬高阈值
        "cost_p75": float(invested["花费"].quantile(0.75)) if len(invested) else float("nan"),
        "clk_p75": float(invested["点击量"].quantile(0.75)) if len(invested) else float("nan"),
        "imp_p75": float(invested["曝光量"].quantile(0.75)) if len(invested) else float("nan"),
        "zero_lead_cost_ratio": zero_lead_cost / total_cost if total_cost else 0.0,
    }


def _ctr_status(df: pd.DataFrame, std) -> pd.Series:
    """判定每行点击率相对基准的状态：达标(above) / 显著偏低(below) / 未知(unknown)。

    用途：CTR 达标说明流量端（主图/标题）已经有效，问题在转化端，
    此时严禁再建议改主图或改标题。把这个结论逐行喂给 AI，比让 AI 自己判断可靠。
    """
    n = len(df)
    if std is None:
        return pd.Series(["unknown"] * n, index=df.index)
    bench = std.get("ctr") if hasattr(std, "get") else None
    if bench is None or not np.isfinite(bench) or bench <= 0:
        return pd.Series(["unknown"] * n, index=df.index)
    low_ratio = std.low_ratio("ctr")[0] if hasattr(std, "low_ratio") else 0.7
    ctr = df["CTR_计算"]
    status = pd.Series("unknown", index=df.index, dtype="object")
    status[ctr.notna()] = np.where(ctr[ctr.notna()] < bench * low_ratio, "below", "above")
    return status


def ad_rows_for_llm(df: pd.DataFrame, std=None) -> list[dict]:
    """组装送给 AI 的精简广告字段。std 可选，用于逐行给出 ctr_status。"""
    sub = pd.DataFrame({
        "product_id": df["产品ID"].astype(str),
        "name": truncate_series(df["商品信息"], MAX_NAME_LEN),
        "cost": df["花费"].round(2),
        "leads": df["商机量"].astype(int),
        "cost_per_lead": df["单次商机成本_计算"].round(2),
        "clicks": df["点击量"].astype(int),
        "impressions": df["曝光量"].astype(int),
        "ctr": df["CTR_计算"].round(4),
        "lead_rate_calc": df["商机转化率_计算"].round(4),
        "invested": df["是否投放"],
    })
    if "核心品类词" in df.columns:
        sub["core_category"] = df["核心品类词"]
    if std is not None:
        sub["ctr_status"] = _ctr_status(df, std)
    if "_local_tag" in df.columns:
        sub["local_tag"] = df["_local_tag"]
    return json_safe(sub.astype(object).where(pd.notna(sub), None).to_dict(orient="records"))


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """CSV 兜底导出（Excel 引擎异常时使用）。"""
    return df.to_csv(index=False).encode("utf-8-sig")
