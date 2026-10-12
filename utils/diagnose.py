"""本地规则预筛 / AI 基准对象 / 分批调度 / 结果解析与合并。

对应 PRD §3.4.3、§3.5.1、§3.5.2、§3.5.4、§4.5.1、§6.7。

⚠️ 设计红线（PRD §3.4）：
本文件只「执行 AI 给的阈值」，不自创任何行业基准值。
DEFAULT_LOW_RATIO 仅在 AI 未返回 thresholds 时使用，且 UI 必须标注"系统默认"。
"""
from __future__ import annotations

import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import (
    DEFAULT_LOW_RATIO, MAX_ROWS_PER_CALL, MAX_ROWS_PER_CALL_LONGNAME,
    MAX_SUGGESTION_LEN, AI_FULL_LIMIT, AI_TOP_K, CONCURRENCY, RETRY_SLEEP, RETRY_TIMES,
    TAG_HIGH_IMP_LOW_CTR, TAG_HIGH_CLK_LOW_OPP, TAG_GOOD, TAG_NORMAL, PRODUCT_TAGS,
    AD_TAG_BURN, AD_TAG_GOLD, AD_TAG_TRAFFIC, AD_TAG_USELESS, AD_TAG_NORMAL, AD_TAGS,
    AD_TAG_TESTING, AD_TAG_NOT_RUNNING, AD_MIN_COST_FOR_JUDGE, AD_MIN_CLICKS_FOR_JUDGE,
    CATEGORY_LEXICON,
    COL_TAG, COL_SOURCE, COL_REASON, COL_SUGGESTION, COL_TAG_PLAIN,
    SOURCE_AI, SOURCE_LOCAL, SOURCE_LOCAL_AI_FAIL, SOURCE_LOCAL_AI_ERROR,
)
from utils.clean import truncate, json_safe

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None


# ==========================================================
# 1. AI 行业基准对象（PRD §3.4.3）
# ==========================================================

# 指标的别名 -> 内部识别用的规范 key。用于兼容 AI 自由发挥的 key 命名。
_METRIC_ALIASES = {
    "ctr": ["ctr", "ctr_rate", "click_rate", "click_through_rate", "搜索点击率", "点击率"],
    "inquiry_rate": ["inquiry_rate", "inq_rate", "inquiry_conversion", "询盘率", "询盘转化率"],
    # ★ 核心判据：商机率（商机数 = 询盘个数 + TM咨询人数），兼容 AI 自由发挥的 key 命名
    "opportunity_rate": ["opportunity_rate", "opp_rate", "opportunity_conversion",
                         "商机率", "商机转化率", "商机转化"],
    "cpc": ["cpc", "cost_per_click", "平均点击成本", "单次点击成本"],
    "cost_per_lead": ["cost_per_lead", "cpl", "商机成本", "平均商机成本", "单次商机成本"],
    "lead_rate": ["lead_rate", "lead_conversion", "商机转化率"],
}


@dataclass
class IndustryStandard:
    """AI 联网自主调研出的行业基准。所有数值来源均为 AI 返回，本地不发明任何值。"""

    category: str = ""
    market: str = ""
    metrics: list = field(default_factory=list)
    thresholds: dict = field(default_factory=dict)
    market_notes: list = field(default_factory=list)
    overall_confidence: str = ""
    raw_text: str = ""
    source_urls: list = field(default_factory=list)
    searched: bool = False
    search_count: int = 0
    created_at: str = ""
    search_summary: str = ""            # AI 自述本次检索了什么、是否找到公开数据
    ok: bool = False                    # 是否成功解析
    degraded: bool = False              # 是否已降级为「店铺内部相对对比」
    degrade_reason: str = ""
    salvaged: bool = False              # 是否靠「抢救子对象」得到的部分指标（UI 需提示）

    # -------- 取值 --------
    def _index(self) -> dict:
        idx = {}
        for m in self.metrics or []:
            if not isinstance(m, dict):
                continue
            key = str(m.get("key", "")).strip().lower()
            name = str(m.get("name", "")).strip().lower()
            if key:
                idx[key] = m
            if name:
                idx.setdefault(name, m)
        return idx

    def find(self, canonical: str) -> dict | None:
        """按规范名找指标；先精确匹配 key/name，再按别名包含匹配。"""
        idx = self._index()
        for a in _METRIC_ALIASES.get(canonical, [canonical]):
            if a.lower() in idx:
                return idx[a.lower()]
        # 模糊：key/name 中包含别名片段
        for k, m in idx.items():
            for a in _METRIC_ALIASES.get(canonical, [canonical]):
                if a.lower() in k:
                    return m
        return None

    def _normalize_unit(self, m: dict):
        """把 AI 给的指标值归一到「可与本地计算值直接比较」的量纲。

        本地点击率/询盘率均为小数（0.035），而 AI 输出的 unit="%" 时 value=3.5，
        因此需要 /100；unit 为 元 时不处理。
        """
        v = m.get("value")
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        unit = str(m.get("unit", "")).strip().lower()
        if unit in ("%", "百分比", "pct", "percent"):
            v = v / 100.0
        elif unit in ("‱",):
            v = v / 10000.0
        if not np.isfinite(v):
            return None
        return v

    def get(self, key: str, default=None):
        """按规范 key 取基准值（已做单位归一）。取不到返回 default。"""
        m = self.find(key)
        if not m:
            return default
        v = self._normalize_unit(m)
        return default if v is None else v

    def has(self, key: str) -> bool:
        return self.find(key) is not None

    def low_ratio(self, key: str, default: float = DEFAULT_LOW_RATIO) -> tuple[float, bool]:
        """取 AI 建议的低阈值比例。返回 (值, 是否来自 AI)。

        第二个返回值用于 UI 标注阈值来源（AI 建议 / 系统默认）。
        """
        raw = self.thresholds.get(f"{key}_low_ratio") if self.thresholds else None
        if raw is None:
            raw = self.thresholds.get(f"{key}_ratio") if self.thresholds else None
        if raw is None:
            return default, False
        try:
            v = float(raw)
            if not np.isfinite(v) or v <= 0:
                return default, False
            return v, True
        except (TypeError, ValueError):
            return default, False

    def high_ratio(self, key: str, default: float = 1.5) -> tuple[float, bool]:
        """取 AI 建议的高阈值倍数（如 cost_high_ratio）。"""
        raw = self.thresholds.get(f"{key}_high_ratio") if self.thresholds else None
        if raw is None:
            return default, False
        try:
            v = float(raw)
            if not np.isfinite(v) or v <= 0:
                return default, False
            return v, True
        except (TypeError, ValueError):
            return default, False

    def threshold_note(self) -> str:
        return str((self.thresholds or {}).get("note", "")).strip()

    def to_text(self) -> str:
        """导出可读文本（写入 Excel 的行业基准 Sheet）。"""
        lines = [
            f"类目：{self.category}",
            f"目标市场：{self.market}",
            f"整体置信度：{self.overall_confidence}",
            f"本次联网检索次数：{self.search_count}",
            "",
            "【基准指标】",
        ]
        for m in self.metrics or []:
            if not isinstance(m, dict):
                continue
            rng = m.get("range") or []
            rng_txt = f"（区间 {rng[0]}~{rng[1]}）" if len(rng) == 2 else ""
            lines.append(
                f"- {m.get('name', m.get('key', ''))}：{m.get('value')}{m.get('unit', '')}"
                f"{rng_txt}｜依据：{m.get('basis', '')}｜置信度：{m.get('confidence', '')}"
            )
            if m.get("source"):
                lines.append(f"    来源：{m.get('source')}")
        if self.thresholds:
            lines += ["", "【判定阈值】", json.dumps(self.thresholds, ensure_ascii=False, indent=2)]
        if self.market_notes:
            lines += ["", "【市场行情洞察】"] + [f"- {x}" for x in self.market_notes]
        if self.source_urls:
            lines += ["", "【检索来源】"] + [f"- {u}" for u in self.source_urls]
        note = self.threshold_note()
        if note:
            lines += ["", f"阈值依据：{note}"]
        lines.append("⚠️ 本基准由 AI 联网检索与自主判断生成，仅供参考，请结合店铺实际经营阶段判断。")
        return "\n".join(lines)


EMPTY_STANDARD = IndustryStandard(ok=False, degraded=True,
                                  degrade_reason="尚未进行行业调研")


# ==========================================================
# 2. JSON 解析（PRD §3.4.3 / §3.5.4）
# ==========================================================

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)
_JSON_ARR_RE = re.compile(r"\[.*\]", re.S)


def repair_json_text(text: str) -> str:
    """常见非法 JSON 修补：去掉代码块围栏、修掉尾逗号、全角冒号等。"""
    if not text:
        return ""
    t = text.strip()
    # 去掉 ```json ... ```
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    # 全角标点
    t = t.replace("：", ": ").replace("，", ",").replace("“", '"').replace("”", '"')
    # NaN / Infinity -> null
    t = re.sub(r"\bNaN\b", "null", t)
    t = re.sub(r"\bInfinity\b", "null", t)
    t = re.sub(r"-Infinity", "null", t)
    # 尾逗号
    t = re.sub(r",\s*([}\]])", r"\1", t)
    # 模型偶尔用单引号当定界符写键名：,'confidence":"高" -> ,"confidence":"高"
    # 否则单引号会吞掉后面的内容，导致整个 JSON 失效
    t = re.sub(r"(?<=[,{\[])\s*'([\w\u4e00-\u9fff]+)\"\s*:", r'"\1":', t)
    return t.strip()


def salvage_objects(text: str) -> list[dict]:
    """抢救嵌套的 {...} 子对象。

    整份 JSON 因为某处语法错误解析失败时，里面各个指标对象往往仍是合法的。
    这里按括号配对收集「嵌套层级的」对象（start 时栈深 >= 1），逐个尝试解析，
    能解析的都留下，避免因为一处笔误丢掉全部基准。
    """
    if not text:
        return []
    t = repair_json_text(text)
    stack: list[int] = []
    buf: list[str] = []
    in_str, esc = False, False
    out: list[dict] = []
    for ch in t:
        if in_str:
            buf.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            buf.append(ch)
            continue
        if ch == "{":
            stack.append(len(buf))
            buf.append(ch)
            continue
        if ch == "}":
            buf.append(ch)
            if stack:
                start = stack.pop()
                if start > 0:  # 嵌套对象（不是最外层）
                    frag = "".join(buf[start:]).strip().rstrip(",")
                    try:
                        obj = json.loads(frag)
                    except Exception:
                        try:
                            obj = json.loads(repair_json_text(frag))
                        except Exception:
                            obj = None
                    if isinstance(obj, dict):
                        out.append(obj)
                    # 用占位符替换，避免干扰外层配对
                    buf = buf[:start] + ["0"]
            continue
        if stack:
            buf.append(ch)
    return out


def parse_json_object(text: str) -> dict | None:
    """从 AI 返回中解析 JSON 对象（行业标准）。"""
    if not text:
        return None
    for candidate in (text, repair_json_text(text)):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        m = _JSON_OBJ_RE.search(candidate)
        if m:
            for try_text in (m.group(0), repair_json_text(m.group(0))):
                try:
                    obj = json.loads(try_text)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    continue
    return None


def parse_json_array(text: str) -> list | None:
    """PRD §3.5.4 原样实现 + 修补增强。"""
    m = _JSON_ARR_RE.search(text or "", re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            try:
                return json.loads(repair_json_text(m.group(0)))
            except Exception:
                pass
    arr = parse_json_array_loose(text)
    return arr


def parse_json_array_loose(text: str) -> list | None:
    """兜底：逐段抽取 {...} 对象，容忍 AI 输出不规范（如缺少外层 []）。"""
    if not text:
        return None
    t = repair_json_text(text)
    objs = []
    depth, buf, in_str, esc = 0, [], False, False
    for ch in t:
        if in_str:
            buf.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            buf.append(ch)
            continue
        if ch == "{":
            depth += 1
            buf.append(ch)
        elif ch == "}":
            depth -= 1
            buf.append(ch)
            if depth == 0:
                try:
                    objs.append(json.loads("".join(buf)))
                except Exception:
                    pass
                buf = []
        elif depth > 0:
            buf.append(ch)
    return objs or None


# ---- 整份 JSON 失效时的字段级抢救 ----

def _regex_scalar(text: str, key: str) -> str:
    """从残缺 JSON 里正则提取一个字符串字段。"""
    m = re.search(rf'"{key}"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text or "")
    return m.group(1).strip() if m else ""


def _balanced_slice(text: str, start: int, open_ch: str, close_ch: str) -> str | None:
    """从 start 处（应指向开括号）截取括号配对的片段。"""
    depth, i, in_str, esc = 0, start, False, False
    while i < len(text):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
        i += 1
    return None


def _extract_after_key(text: str, key: str, open_ch: str, close_ch: str):
    m = re.search(rf'"{key}"\s*:\s*\{open_ch}', text or "")
    if not m:
        return None
    frag = _balanced_slice(text, m.end() - 1, open_ch, close_ch)
    if not frag:
        return None
    for attempt in (frag, repair_json_text(frag)):
        try:
            val = json.loads(attempt)
        except Exception:
            continue
        if isinstance(val, dict) and open_ch == "{":
            return val
        if isinstance(val, list) and open_ch == "[":
            return val
    return None


def _extract_object_after_key(text: str, key: str) -> dict | None:
    v = _extract_after_key(text, key, "{", "}")
    return v if isinstance(v, dict) else None


def _extract_array_after_key(text: str, key: str) -> list | None:
    v = _extract_after_key(text, key, "[", "]")
    return v if isinstance(v, list) else None


def build_industry_standard(text: str, *, fallback: dict | None = None,
                            searched: bool = False, search_count: int = 0,
                            category_default: str = "", market_default: str = "") -> IndustryStandard:
    """把 AI 返回的文本解析成 IndustryStandard。

    解析失败 -> ok=False 且保留原文（PRD §3.4.3 要求展示原文而非静默降级）。
    """
    from datetime import datetime
    obj = parse_json_object(text)
    std = IndustryStandard(
        raw_text=text or "",
        searched=searched,
        search_count=search_count,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    if obj is None:
        # 兜底一：抢救嵌套的指标对象（整份 JSON 因个别笔误失效时仍保住可用指标）
        salvaged = [o for o in salvage_objects(text)
                    if (o.get("key") or o.get("name"))
                    and isinstance(o.get("value"), (int, float))]
        if salvaged:
            std.metrics = salvaged
            std.salvaged = True
            std.ok = True
            std.degrade_reason = "AI 返回的 JSON 存在语法错误，已抢救出部分指标"
        else:
            # 兜底二：PRD §3.4.3 正则提取「点击率/询盘率」后的百分数
            picked = []
            for label, key in (("点击率", "ctr"), ("询盘率", "inquiry_rate")):
                m = re.search(rf"{label}[^0-9\-+]{{0,12}}([0-9]+(?:\.[0-9]+)?)\s*%", text or "")
                if m:
                    picked.append({
                        "key": key, "name": label, "value": float(m.group(1)), "unit": "%",
                        "range": [], "basis": "经验推断",
                        "source": "从返回文本正则提取（模型未按格式输出 JSON）", "confidence": "低",
                    })
            if picked:
                std.metrics = picked
                std.ok = True
            else:
                std.degraded = True
                std.degrade_reason = "AI 返回无法解析为 JSON，行业基准不可用"

        # 补齐顶层标量字段与阈值
        std.category = _regex_scalar(text, "category") or category_default
        std.market = _regex_scalar(text, "market") or market_default
        std.overall_confidence = _regex_scalar(text, "overall_confidence") or ""
        std.thresholds = _extract_object_after_key(text, "thresholds") or {}
        std.market_notes = _extract_array_after_key(text, "market_notes") or []
        return std

    std.category = str(obj.get("category") or category_default or "")
    std.market = str(obj.get("market") or market_default or "")
    std.metrics = obj.get("metrics") or []
    std.thresholds = obj.get("thresholds") or {}
    std.market_notes = [str(x) for x in (obj.get("market_notes") or [])]
    std.overall_confidence = str(obj.get("overall_confidence") or "")
    std.search_summary = str(obj.get("search_summary") or "")
    urls = obj.get("answer_source_urls") or []
    seen, clean = set(), []
    for u in urls:
        u = str(u).strip()
        if u.startswith("http") and u not in seen:
            seen.add(u)
            clean.append(u)
    std.source_urls = clean
    std.ok = bool(std.metrics) or bool(std.thresholds)

    if fallback:
        # 类目 / 市场由用户指定或 AI 未给出时补齐
        std.category = std.category or fallback.get("category", "")
        std.market = std.market or fallback.get("market", "")
    if not std.metrics and not std.thresholds:
        std.degraded = True
        std.degrade_reason = "AI 未返回任何基准指标或判定阈值"
    return std


# ==========================================================
# 3. 本地规则（模块一，PRD §3.5.1）+ 分位数退化处理
# ==========================================================

def _store_fallback_ctr(df: pd.DataFrame) -> float:
    """行业基准缺失时的降级值：全店加权点击率（有曝光的行）。"""
    sub = df[df["搜索曝光次数"] > 0]
    if not len(sub):
        return float("nan")
    return float(sub["搜索点击次数"].sum() / sub["搜索曝光次数"].sum())


def _store_fallback_inq(df: pd.DataFrame) -> float:
    """降级值：全店加权询盘率（有点击的行）。"""
    sub = df[df["搜索点击次数"] > 0]
    if not len(sub):
        return float("nan")
    return float(sub["询盘个数"].sum() / sub["搜索点击次数"].sum())


def _store_fallback_opp(df: pd.DataFrame) -> float:
    """降级值：全店加权商机率（有点击的行）。商机数 = 询盘个数 + TM咨询人数。"""
    col = "商机数" if "商机数" in df.columns else "询盘个数"
    sub = df[df["搜索点击次数"] > 0]
    if not len(sub):
        return float("nan")
    return float(sub[col].sum() / sub["搜索点击次数"].sum())


def local_tag_product(df: pd.DataFrame, std: IndustryStandard, summary: dict) -> dict:
    """本地规则预筛（PRD §3.5.1）。返回 {'series': Series, 'info': dict}。

    阈值全部取自 AI 的 metrics / thresholds；AI 未给则用 DEFAULT_LOW_RATIO 并在 info 标注。
    行业基准缺失时降级为「店铺内部相对对比」，并在 info.degraded 标注。
    """
    info = {"thr_source": {}, "used_default_ratio": [], "degraded": False, "notes": []}

    ctr_ind, ctr_src = std.get("ctr"), std.has("ctr")
    # ★ 核心转化判据：商机率（不是询盘率——买家多数走 TM 而非询盘）
    opp_ind, opp_src = std.get("opportunity_rate"), std.has("opportunity_rate")

    # --- 行业基准缺失 -> 降级为店铺内部相对对比（PRD §3.5.1 规则 3）---
    if ctr_ind is None or (isinstance(ctr_ind, float) and np.isnan(ctr_ind)):
        ctr_ind = _store_fallback_ctr(df)
        info["degraded"] = True
        ctr_src = False
    if opp_ind is None or (isinstance(opp_ind, float) and np.isnan(opp_ind)):
        opp_ind = _store_fallback_opp(df)
        info["degraded"] = True
        opp_src = False

    ctr_low, ctr_low_ai = std.low_ratio("ctr")
    opp_low, opp_low_ai = std.low_ratio("opportunity_rate")
    info["thr_source"] = {"ctr": "AI 建议" if ctr_low_ai else "系统默认",
                          "opportunity_rate": "AI 建议" if opp_low_ai else "系统默认"}
    if not ctr_low_ai:
        info["used_default_ratio"].append("ctr")
    if not opp_low_ai:
        info["used_default_ratio"].append("opportunity_rate")

    exp_p50 = summary.get("exp_p50", 0.0)
    clk_p50 = summary.get("clk_p50", 0.0)

    # --- 分位数退化处理（实测：冷启动店铺 点击P50=0，「>=P50」恒真）---
    exp_thr = float(exp_p50)
    clk_thr = float(clk_p50)
    if exp_thr < 1:
        exp_thr = 1.0
        info["notes"].append(
            f"全店曝光中位数为 {exp_p50:g}，「高曝光」判定已改为「有曝光（≥1）」，避免规则退化")
    if clk_thr < 1:
        clk_thr = 1.0
        info["notes"].append(
            f"全店点击中位数为 {clk_p50:g}，「高点击」判定已改为「有点击（≥1）」，避免规则退化")

    imp = df["搜索曝光次数"]
    clk = df["搜索点击次数"]
    opp = df["商机数"] if "商机数" in df.columns else df["询盘个数"]
    ctr = df["点击率_计算"]
    opp_rate = df["商机率_计算"] if "商机率_计算" in df.columns else df["询盘率_计算"]

    has_ctr = bool(ctr_ind is not None and np.isfinite(ctr_ind))
    has_opp = bool(opp_ind is not None and np.isfinite(opp_ind))

    # 条件 1：高曝光低点击
    c1 = (imp >= exp_thr)
    c1 = c1 & (ctr.notna()) & (ctr < ctr_ind * ctr_low) if has_ctr else pd.Series(False, index=df.index)

    # 条件 2：高点击低商机（★ 原「高点击低询盘」，按买家实际走 TM 的路径改为商机）
    if has_opp:
        c2 = (clk >= clk_thr) & ((opp == 0) | ((opp_rate.notna()) & (opp_rate < opp_ind * opp_low)))
    else:
        c2 = (clk >= clk_thr) & (opp == 0)

    # 条件 3：表现良好
    if has_ctr and has_opp:
        c3 = (ctr.notna()) & (ctr >= ctr_ind) & (opp_rate.notna()) & (opp_rate >= opp_ind)
    else:
        c3 = pd.Series(False, index=df.index)

    tags = np.select(
        [c1.to_numpy(), c2.to_numpy(), c3.to_numpy()],
        [TAG_HIGH_IMP_LOW_CTR, TAG_HIGH_CLK_LOW_OPP, TAG_GOOD],
        default=TAG_NORMAL,
    )
    series = pd.Series(tags, index=df.index, dtype="object")
    # 曝光为 0 的行样本不足，不参与任何异常判定（PRD T9）
    series = series.mask(df["搜索曝光次数"] == 0, TAG_NORMAL)
    return {"series": series, "info": info}


# ==========================================================
# 4. 本地规则（模块二，PRD §4.5.1）
# ==========================================================

def local_tag_ad(df: pd.DataFrame, std: IndustryStandard, summary: dict) -> dict:
    """广告本地规则（PRD §4.5.1）。返回 {'series': Series, 'info': dict}。

    优先级自上而下命中即停；花费 == 0 直接判 🟢 正常。
    M = 有效投放行数，M<5 时改用「行业基准 + AI 阈值」而非分位数。
    """
    info = {"mode": "", "thr_source": {}, "used_default_ratio": [], "degraded": False, "notes": []}
    invested = df["是否投放"]
    M = int(invested.sum())

    if M == 0:
        # ★ 用户反馈：花费为 0 是"根本没投放"，不能算"正常"
        return {"series": pd.Series(AD_TAG_NOT_RUNNING, index=df.index, dtype="object"),
                "info": {"mode": "M=0",
                         "notes": ["本次数据无任何实际投放花费，全部标记为「⚫ 未投放（待激活）」"],
                         "thr_source": {}, "used_default_ratio": [], "degraded": False}}

    cpl_ind = std.get("cost_per_lead")
    ctr_ind = std.get("ctr")
    lr_ind = std.get("lead_rate")

    if cpl_ind is None or (isinstance(cpl_ind, float) and np.isnan(cpl_ind)):
        sub = df[df["单次商机成本_计算"].notna()]
        cpl_ind = float(sub["单次商机成本_计算"].median()) if len(sub) else float("inf")
        info["degraded"] = True
    if ctr_ind is None or (isinstance(ctr_ind, float) and np.isnan(ctr_ind)):
        ctr_ind = float(df["CTR_计算"].median(skipna=True)) if df["CTR_计算"].notna().any() else 0.0
        info["degraded"] = True
    if lr_ind is None or (isinstance(lr_ind, float) and np.isnan(lr_ind)):
        lr_ind = float(df["商机转化率_计算"].median(skipna=True)) if df["商机转化率_计算"].notna().any() else 0.0
        info["degraded"] = True

    ctr_low, ctr_ai = std.low_ratio("ctr")
    lr_low, lr_ai = std.low_ratio("lead_rate")
    cost_high, cost_ai = std.high_ratio("cost")
    info["thr_source"] = {"ctr": "AI 建议" if ctr_ai else "系统默认",
                          "lead_rate": "AI 建议" if lr_ai else "系统默认",
                          "cost": "AI 建议" if cost_ai else "系统默认"}
    info["used_default_ratio"] = [k for k, v in (("ctr", ctr_ai), ("lead_rate", lr_ai)) if not v]

    cost = df["花费"]
    # ★ 商机口径对齐：leads 用「商机数 = 商机量 + TM咨询量」
    leads = df["商机数"] if "商机数" in df.columns else df["商机量"]
    clicks = df["点击量"]
    imps = df["曝光量"]
    ctr = df["CTR_计算"]
    lr = df["商机转化率_计算"]
    cpl = df["单次商机成本_计算"]

    def _has(x):
        return x is not None and np.isfinite(x) if isinstance(x, float) else x is not None

    if M >= 5:
        info["mode"] = f"分位数模式（M={M}≥5）"
        cost_p75 = float(cost[invested].quantile(0.75))
        clk_p75 = float(clicks[invested].quantile(0.75))
        imp_p75 = float(imps[invested].quantile(0.75))
        c1 = invested & (cost >= cost_p75) & (leads == 0)
        c2 = invested & (leads > 0) & cpl.notna() & (cpl <= cpl_ind) if _has(cpl_ind) else \
            invested & (leads > 0)
        c3 = invested & (clicks >= clk_p75) & lr.notna() & (lr < lr_ind * lr_low) if _has(lr_ind) else \
            pd.Series(False, index=df.index)
        c4 = invested & (imps >= imp_p75) & ctr.notna() & (ctr < ctr_ind * ctr_low) if _has(ctr_ind) else \
            pd.Series(False, index=df.index)
    else:
        info["mode"] = f"基准模式（M={M}<5，样本太小改用行业基准）"
        info["notes"].append(
            f"有效投放行数仅 {M} 条，分位数失真，已改用「行业基准 × AI 阈值」判定")
        c1 = invested & (leads == 0)
        c2 = invested & (leads > 0) & cpl.notna() & (cpl <= cpl_ind) if _has(cpl_ind) else \
            invested & (leads > 0)
        c3 = invested & lr.notna() & (lr < lr_ind * lr_low) if _has(lr_ind) else \
            pd.Series(False, index=df.index)
        c4 = invested & ctr.notna() & (ctr < ctr_ind * ctr_low) if _has(ctr_ind) else \
            pd.Series(False, index=df.index)

    # ★★ 两条强制规则（用户反馈，优先级高于上面所有定性判断）★★
    # 1) 花费 = 0            -> ⚫ 未投放（待激活）：根本没投放，不算"正常"
    # 2) 花费 < 50 且 点击 < 10 -> 🧪 测试中（样本不足）：B2B 冷启动小额测词属正常，不判烧钱品
    c_not_running = ~invested
    c_testing = invested & (cost < AD_MIN_COST_FOR_JUDGE) & (clicks < AD_MIN_CLICKS_FOR_JUDGE)
    n_testing = int(c_testing.sum())

    tags = np.select(
        [c_not_running.to_numpy(), c_testing.to_numpy(),
         c1.to_numpy(), c2.to_numpy(), c3.to_numpy(), c4.to_numpy()],
        [AD_TAG_NOT_RUNNING, AD_TAG_TESTING,
         AD_TAG_BURN, AD_TAG_GOLD, AD_TAG_TRAFFIC, AD_TAG_USELESS],
        default=AD_TAG_NORMAL,
    )
    series = pd.Series(tags, index=df.index, dtype="object")
    if n_testing:
        info["notes"].append(
            f"其中 {n_testing} 条花费 < {AD_MIN_COST_FOR_JUDGE:g} 元且点击 < {AD_MIN_CLICKS_FOR_JUDGE} 次，"
            f"已标记为「🧪 测试中（样本不足）」，不参与烧钱品等定性判断"
        )
    info["notes"].append(
        f"花费为 0 的 {int(c_not_running.sum())} 条已标记为「⚫ 未投放（待激活）」（不再算作正常）"
    )
    return {"series": series, "info": info}


# ==========================================================
# 5. 分批策略（PRD §3.5.2 / §6.7）
# ==========================================================

def calc_batches(n_rows: int, max_rows: int = MAX_ROWS_PER_CALL) -> int:
    """动态批数（禁止硬编码行数）。"""
    return max(math.ceil(n_rows / max_rows), 0)


def plan_rows_per_call(names: pd.Series) -> int:
    """名称过长时自适应降低单批行数（PRD §3.5.2）。"""
    try:
        avg_len = float(names.astype(str).str.len().mean())
    except Exception:
        return MAX_ROWS_PER_CALL
    return MAX_ROWS_PER_CALL_LONGNAME if avg_len > 90 else MAX_ROWS_PER_CALL


def select_ai_rows(df: pd.DataFrame, score: pd.Series | None, top_k: int,
                   ai_full_limit: int = AI_FULL_LIMIT) -> tuple[list, str]:
    """决定哪些行送 AI。返回 (行下标列表, 模式标记)。

    N <= AI_FULL_LIMIT -> full（全量送 AI）
    N  > AI_FULL_LIMIT -> hybrid（本地规则全量 + AI 精修 Top-K）
    """
    n = len(df)
    if n <= ai_full_limit:
        return list(range(n)), "full"
    k = max(min(int(top_k), n), 0)
    if score is not None and len(score) == n:
        order = list(score.sort_values(ascending=False).index[:k])
    else:
        order = list(df.index[:k])
    return order, "hybrid"


def product_priority_score(df: pd.DataFrame, local_tags: pd.Series) -> pd.Series:
    """产品 Top-K 排序权重：曝光量 × 异常程度（PRD §3.5.2）。"""
    severity = local_tags.map({
        TAG_HIGH_IMP_LOW_CTR: 3.0, TAG_HIGH_CLK_LOW_OPP: 3.0, TAG_GOOD: 0.5, TAG_NORMAL: 1.0,
    }).fillna(1.0).astype(float)
    return df["搜索曝光次数"].astype(float) * severity


def ad_priority_order(df: pd.DataFrame) -> pd.Series:
    """广告 Top-K 排序权重：花费降序为主、商机量升序次之（PRD §4.5.2）。"""
    tmp = pd.DataFrame({"cost": df["花费"].astype(float),
                        "leads": df["商机量"].astype(float)})
    return tmp.sort_values(["cost", "leads"], ascending=[False, True]).index


def chunk_list(idx: list, size: int) -> list[list]:
    return [idx[i:i + size] for i in range(0, len(idx), size)]


# ==========================================================
# 6. 通用分批调度（PRD §6.7）
# ==========================================================

class _Throttle(Exception):
    """触发限流，需要降低并发。"""


def _state():
    """session_state 会在 Streamlit 每次交互重置未赋值键，这里做安全读取。"""
    return st.session_state if st is not None else None


def run_batches(
    idx_rows: list,
    df_all: pd.DataFrame,
    task,
    state_key: str,
    max_rows: int = MAX_ROWS_PER_CALL,
    concurrency: int = CONCURRENCY,
    progress_cb=None,
    stop_flag_key: str = "",
    max_waves: int | None = None,
) -> dict[int, list]:
    """max_waves: 本次最多执行几「波」。

    用途：Streamlit 同步循环期间浏览器收不到中间渲染，必须每波结束 st.rerun()
    才能看到实时进度 / 响应「中止」。调用方传 max_waves=1 逐波推进，
    已完成批次从 session_state 恢复（天然支持断点续跑）。
    """
    """通用分批执行器（两 Tab 复用）。

    task(chunk_df, batch_idx) -> (ok: bool, payload)  payload 为结果列表或错误信息
    返回 {batch_idx: [ {...}, ... ]}

    特性（PRD §3.5.2）：
    - 断点续跑：已完成批次存 session_state[f"{state_key}_batch_results"]
    - 中止：session_state[f"{state_key}_stop"] 为 True 时停止后续批次
    - 失败重试 1 次 + 429 自动退避并降并发至 1
    """
    s = _state()
    key_results = f"{state_key}_batch_results"
    key_failed = f"{state_key}_failed_batches"
    key_stop = stop_flag_key or f"{state_key}_stop"

    key_finished = f"{state_key}_finished"
    key_attempts = f"{state_key}_attempts"
    max_attempts = RETRY_TIMES + 1
    if s is not None:
        results = s.get(key_results, {})
        failed = list(s.get(key_failed, []))
        attempts = dict(s.get(key_attempts, {}))
        s[key_finished] = False
    else:
        results, failed, attempts = {}, [], {}

    batches = chunk_list(list(idx_rows), max_rows)
    total = len(batches)

    def _stopped() -> bool:
        return bool(s.get(key_stop, False)) if s is not None else False

    def _persist(finished: bool = False):
        if s is not None:
            s[key_results] = results
            s[key_failed] = failed
            s[key_attempts] = attempts
            s[key_finished] = finished

    done_count = 0
    cur_conc = max(int(concurrency), 1)
    throttle_hit = False
    waves_done = 0

    i = 0
    while i < total:
        if _stopped():
            break
        if max_waves is not None and waves_done >= max_waves:
            break
        wave = []
        while len(wave) < cur_conc and i < total:
            bidx = i
            settled = bidx in results and bidx not in failed
            if settled:
                done_count += 1
            elif attempts.get(bidx, 0) >= max_attempts:
                # 已耗尽重试次数：判定为永久失败，避免每波无限重试
                if bidx not in failed:
                    failed.append(bidx)
            else:
                wave.append(bidx)
                attempts[bidx] = attempts.get(bidx, 0) + 1
            i += 1
        if not wave:
            continue

        if _stopped():
            break

        wave_results: dict[int, tuple[bool, object]] = {}
        with ThreadPoolExecutor(max_workers=len(wave)) as ex:
            futs = {}
            for bidx in wave:
                chunk_df = df_all.loc[batches[bidx]]
                futs[ex.submit(task, chunk_df, bidx)] = bidx
            for fut in as_completed(futs):
                bidx = futs[fut]
                try:
                    wave_results[bidx] = fut.result()
                except _Throttle as e:
                    wave_results[bidx] = (False, f"THROTTLE:{e}")
                except Exception as e:
                    wave_results[bidx] = (False, f"{type(e).__name__}: {e}")

        for bidx in wave:
            ok, payload = wave_results.get(bidx, (False, "未返回结果"))
            if ok and payload:
                results[bidx] = payload
                if bidx in failed:
                    failed.remove(bidx)
            else:
                err = str(payload)
                if "THROTTLE" in err or "429" in err or "Throttl" in err:
                    throttle_hit = True
                # 失败：自动重试 1 次（PRD §3.5.2(4)）
                time.sleep(RETRY_SLEEP)
                try:
                    ok2, payload2 = task(df_all.loc[batches[bidx]], bidx)
                    if ok2 and payload2:
                        results[bidx] = payload2
                        if bidx in failed:
                            failed.remove(bidx)
                        continue
                except Exception:
                    pass
                if bidx not in failed:
                    failed.append(bidx)
                if bidx in results:
                    results.pop(bidx, None)

        done_count += len(wave)
        waves_done += 1
        if progress_cb:
            try:
                progress_cb(done_count, total, len(failed))
            except Exception:
                pass

        if throttle_hit and cur_conc > 1:
            cur_conc = 1
            throttle_hit = False

    # 是否全部"落定"：每个批次要么有结果，要么已耗尽重试，且未被中止
    all_settled = all((b in results) or (attempts.get(b, 0) >= max_attempts) or (b in failed)
                      for b in range(total)) and not _stopped()
    _persist(all_settled)
    return results


# ==========================================================
# 7. AI 结果合并（PRD §3.5.4）
# ==========================================================

# 注意 ⚪ 的码点是 U+26AA（基本平面），而 🔴🟠🟡🔵 等在 \U00010000-\U0010FFFF（星平面），
# 两者必须都覆盖，否则「⚪ 鸡肋品」剥不干净，Excel 条件格式会匹配失败。
_EMOJI_PREFIX_RE = re.compile(
    r"^[\s\u200D\uFE0F\U00010000-\U0010FFFF\u2600-\u27BF\u2B00-\u2BFF]+"
)


def strip_tag_plain(tag: str) -> str:
    """去掉 emoji 前缀，得到纯文本标签（用于 Excel 条件格式匹配）。"""
    t = str(tag or "")
    return _EMOJI_PREFIX_RE.sub("", t).strip() or t


def normalize_tag(tag: str, allowed: list[str]) -> str:
    """把 AI 返回的标签对齐到枚举；无法识别则回落 ⚪/🟢 正常。"""
    t = str(tag or "").strip()
    for a in allowed:
        if t == a:
            return a
    plain = strip_tag_plain(t)
    for a in allowed:
        if strip_tag_plain(a) == plain:
            return a
    for a in allowed:
        if a.endswith("正常") or a.endswith("鸡肋品"):
            return a if plain in ("正常", "鸡肋品") else a
    return allowed[-1]


# 命中"改/换/重做/优化/调整/更新/替换 + 主图/标题"的整个分句
_FORBID_MAIN_RE = re.compile(
    r"[^；;。]*?(?:改|换|重做|优化|调整|更新|替换|重构|重写)\s*(?:主图|标题)[^；;。]*?[；;。]?"
)


def strip_forbidden_clauses(text, forbid_regex=_FORBID_MAIN_RE, fallback: str = "") -> str:
    """剥离建议中包含「改主图 / 改标题」的分句。

    用途：CTR 达标说明流量端没问题，此时给"改主图/改标题"属于方向性错误。
    提示词已明令禁止，但模型仍可能违反，这里做确定性兜底。
    """
    out = forbid_regex.sub("", str(text or ""))
    out = re.sub(r"[；;，,]\s*(?=[；;，,]|$)", "", out).strip(" ；;，,。")
    return out or fallback


def enforce_category_consistency(suggestions: pd.Series, core: pd.Series) -> pd.Series:
    """程序化兜底：建议里若出现与「核心品类词」不同的其它品类词，替换回正确的。

    为什么需要这道兜底：实测仅靠提示词约束，模型仍会把 Shower Chair 改写成 Grab Bar
    （改变产品的基础定义）。逐行喂 core_category + 提示词约束能大幅降低概率，
    但无法 100% 保证；这里做一次确定性替换，确保输出不会误导运营。
    """
    if core is None or suggestions is None or not len(suggestions):
        return suggestions
    out = suggestions.astype("object").fillna("").astype(str)
    core_l = core.astype("object").fillna("").astype(str).str.lower()
    for term in sorted(CATEGORY_LEXICON, key=len, reverse=True):
        mask = (out.str.lower().str.contains(term, regex=False)) & (core_l != term) & (core_l != "")
        if not mask.any():
            continue
        new_vals = []
        for s, c in zip(out.loc[mask], core_l.loc[mask]):
            new_vals.append(re.sub(re.escape(term), c.title(), s, flags=re.I))
        out.loc[mask] = new_vals
    return out


def merge_results(
    df: pd.DataFrame,
    batch_results: dict[int, list],
    id_col: str = "产品ID",
    allowed: list[str] = PRODUCT_TAGS,
    source_col: str = COL_SOURCE,
) -> pd.DataFrame:
    """把 AI 结果左连接回主表；缺失行回退本地规则标签（PRD §3.5.4）。"""
    rows = []
    for _, lst in (batch_results or {}).items():
        if isinstance(lst, list):
            rows.extend([r for r in lst if isinstance(r, dict)])

    out = df.copy()
    ai_map, reason_map, sug_map = {}, {}, {}
    for r in rows:
        pid = str(r.get("product_id", "")).strip()
        if not pid or pid in ("None", "nan"):
            continue
        tag = normalize_tag(r.get("tag"), allowed)
        ai_map[pid] = tag
        reason_map[pid] = str(r.get("reason", "") or "")
        sug_map[pid] = truncate(r.get("suggestion", "") or "", MAX_SUGGESTION_LEN)

    out["_ai_tag"] = out[id_col].astype(str).map(ai_map)
    out[COL_REASON] = out[id_col].astype(str).map(reason_map).fillna("")
    out[COL_SUGGESTION] = out[id_col].astype(str).map(sug_map).fillna("")

    # 兜底：确保建议不会把商品改成另一个品类
    if "核心品类词" in out.columns:
        out[COL_SUGGESTION] = enforce_category_consistency(out[COL_SUGGESTION], out["核心品类词"])

    hit = out["_ai_tag"].notna()
    out[COL_TAG] = np.where(hit, out["_ai_tag"], out["_local_tag"])
    out[source_col] = np.where(hit, SOURCE_AI, SOURCE_LOCAL)
    # 本地规则也没结论 -> ⚪/🟢 正常
    fallback_normal = PRODUCT_TAGS[-1] if allowed is PRODUCT_TAGS else AD_TAGS[-1]
    out[COL_TAG] = pd.Series(out[COL_TAG], index=out.index).replace("", np.nan).fillna(fallback_normal)
    out[COL_TAG_PLAIN] = out[COL_TAG].map(strip_tag_plain)
    return out.drop(columns=["_ai_tag"])


def count_tags(df: pd.DataFrame, allowed: list[str]) -> dict:
    """统计各标签数量。"""
    vc = df[COL_TAG].value_counts().to_dict() if COL_TAG in df.columns else {}
    return {t: int(vc.get(t, 0)) for t in allowed}


def ai_covered_count(df: pd.DataFrame, source_col: str = COL_SOURCE) -> int:
    if source_col not in df.columns:
        return 0
    return int((df[source_col] == SOURCE_AI).sum())
