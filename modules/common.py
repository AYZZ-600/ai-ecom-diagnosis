"""两个 Tab 共用的 UI 组件与 AI 领域函数（避免重复代码）。

对应 PRD §3.4.4、§3.6、§5。
"""
from __future__ import annotations

import io
import json
import math
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    HIGHLIGHT_RED, HIGHLIGHT_GOLD, HIGHLIGHT_GREEN, MAX_STYLE_ROWS,
    DEFAULT_RENDER_ROWS, RENDER_ROWS_OPTIONS, TABLE_HEIGHT, CLASSIFY_SAMPLE,
    COL_TAG, COL_SOURCE, COL_REASON, COL_SUGGESTION, COL_TAG_PLAIN,
    PRODUCT_TAGS, AD_TAGS,
)
from prompts import CATEGORY_IDENTIFY_PROMPT, SYSTEM_CATEGORY, render
from llm import qwen_chat, LLMError, MissingAPIKeyError, extract_text_sources
from utils.diagnose import IndustryStandard, build_industry_standard, strip_tag_plain
from utils.export import to_excel_bytes, report_filename


# ==========================================================
# 1. AI 类目自动识别（PRD §3.4.1）
# ==========================================================

_CAT_LINE = re.compile(
    r"类目[:：]\s*(?P<cat>.+?)\s*[；;]\s*目标市场[:：]\s*(?P<mk>.+?)\s*[；;]\s*"
    r"买家类型[:：]\s*(?P<bt>.+)"
)


def identify_category(titles: list[str]) -> dict:
    """从商品标题样本让 AI 判断类目 / 目标市场。失败返回空字典，不中断流程。"""
    if not titles:
        return {}
    sample = titles[:CLASSIFY_SAMPLE]
    prompt = render(CATEGORY_IDENTIFY_PROMPT, titles="\n".join(f"- {t}" for t in sample))
    try:
        r = qwen_chat(
            messages=[{"role": "system", "content": SYSTEM_CATEGORY},
                      {"role": "user", "content": prompt}],
            enable_search=True,
            temperature=0.2,
            max_tokens=400,
        )
        m = _CAT_LINE.search(r.text.replace("\n", " "))
        if m:
            return {"category": m.group("cat").strip(),
                    "market": m.group("mk").strip(),
                    "buyer_type": m.group("bt").strip(),
                    "search_count": r.search_count}
        return {"raw": r.text, "search_count": r.search_count}
    except LLMError as e:
        return {"error": str(e)}


def build_shop_profile(kind: str, df, summary: dict, meta: dict | None = None,
                       n_titles: int = 8) -> str:
    """把用户上传的本店数据整理成给 AI 的「店铺画像」。

    用途（用户反馈 4）：本店商品偏冷门，纯联网基准往往查不到或不适用，
    因此把本店自身分布一并交给 AI，让它在本店分布与联网基准之间做校准，
    查不到公开数据时以本店分布为相对锚点（并要求标注 basis="本店实际推算"）。
    """
    if df is None or not len(df) or not summary:
        return "（尚未上传数据）"
    meta = meta or {}
    lines = []
    if kind == "product":
        n = summary.get("n_total", len(df))
        lines.append(f"统计口径：本店 {n} 个商品的近期数据（用户本次上传）")
        lines.append(f"- 总曝光 {int(summary.get('impressions', 0)):,}，"
                     f"总点击 {int(summary.get('clicks', 0)):,}，"
                     f"总商机 {int(summary.get('opportunities', 0)):,}"
                     f"（其中询盘 {int(summary.get('inquiries', 0)):,}），"
                     f"总访问 {int(summary.get('visitors', 0) or 0):,}")
        lines.append(f"- 全店加权点击率 {summary.get('ctr_all', float('nan')):.2%}，"
                     f"全店加权商机率 {summary.get('opp_all', float('nan')):.2%}，"
                     f"全店加权询盘率 {summary.get('inq_all', float('nan')):.2%}")
        lines.append(f"- 曝光中位数 {summary.get('exp_p50', 0):.0f}，"
                     f"点击中位数 {summary.get('clk_p50', 0):.0f}，"
                     f"曝光 P75 {summary.get('exp_p75', 0):.0f}，点击 P75 {summary.get('clk_p75', 0):.0f}")
        lines.append(f"- 零商机商品占比 {summary.get('zero_opp_ratio', 0):.1%}；"
                     f"零曝光商品 {meta.get('zero_exposure', 0)} 个，"
                     f"零点击商品 {meta.get('zero_click', 0)} 个")
        title_col = "产品名称"
    else:
        n = summary.get("n", len(df))
        lines.append(f"统计口径：本店 {n} 条推广商品（其中 {summary.get('n_invested', 0)} 条有实际花费）")
        lines.append(f"- 总花费 ￥{summary.get('total_cost', 0):,.2f}，"
                     f"总点击 {int(summary.get('total_clicks', 0)):,}，"
                     f"总曝光 {int(summary.get('total_impressions', 0)):,}，"
                     f"总商机 {int(summary.get('total_leads', 0)):,}")
        lines.append(f"- 整体 CPC ￥{summary.get('cpc_all', float('nan')):.2f}，"
                     f"整体商机转化率 {summary.get('lead_rate_all', float('nan')):.2%}")
        lines.append(f"- 零商机花费占比 {summary.get('zero_lead_cost_ratio', 0):.1%}；"
                     f"零商机点击占比 {summary.get('zero_opp_click_ratio', 0):.1%}")
        title_col = "商品信息"

    if title_col in df.columns:
        titles = [str(t)[:70] for t in df[title_col].dropna().head(n_titles)]
        if titles:
            lines.append("- 商品标题样本（用于判断冷门程度与实际品类）：")
            lines.extend([f"    {i}. {t}" for i, t in enumerate(titles, 1)])
    return "\n".join(lines)


CORRECT_JSON_HINT = """

【重要】你上一次的输出不是合法 JSON，解析失败。请严格只输出一个 JSON 对象：
- 所有键名与字符串值必须用英文双引号 " 包裹，禁止用单引号 ' 作为定界符
- 字符串内部的引号必须转义为 \\"
- 数组最后一项后不要有逗号
- 不要输出任何解释文字、不要 markdown 代码块围栏"""


def research_standard(prompt: str, system: str, category_default: str = "",
                      market_default: str = "", retries: int = 1,
                      spinner_text: str = "AI 正在联网调研行业标准，约需 10~30 秒…"):
    """行业标准调研：调用 → 解析 → 解析失败自动重试（带纠错指令）。

    实测：模型偶尔会输出带语法错误的 JSON（如用单引号写键名），
    单靠修复器不一定救得回，重试一次的成本（约 20s）远低于让用户手工重试。

    返回 (IndustryStandard, LLMResponse|None, error: str|None)
    """
    last_std, last_resp = None, None
    p = prompt
    for attempt in range(retries + 1):
        r = qwen_chat(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": p}],
            enable_search=True, temperature=0.2, max_tokens=2500,
        )
        std = build_industry_standard(
            r.text, searched=r.searched, search_count=r.search_count,
            category_default=category_default, market_default=market_default,
        )
        if not std.source_urls:
            std.source_urls = [u["url"] for u in extract_text_sources(r.text)][:10]
        last_std, last_resp = std, r
        if std.ok:
            return std, r, None
        p = prompt + CORRECT_JSON_HINT
    return last_std, last_resp, (last_std.degrade_reason if last_std else "未知错误")


# ==========================================================
# 2. session_state 工具
# ==========================================================

def sget(key, default=None):
    return st.session_state.get(key, default)


def sset(key, value):
    st.session_state[key] = value


def set_pending(pending_key: str, value) -> None:
    """把要写回控件的值先存到「暂存键」。

    原因：Streamlit 规定控件实例化后不能再写它绑定的 session_state 键，
    否则抛 StreamlitWidgetAlreadyInstantiatedError。
    所以在回调里只能写暂存键，等下一轮渲染、控件实例化之前再落盘。
    """
    st.session_state[pending_key] = value


def apply_pending(pending_key: str, widget_key: str, overwrite: bool = False) -> None:
    """在控件实例化【之前】调用，把暂存值写入控件绑定的 session_state。

    overwrite=False 时仅在控件当前为空时写入（避免覆盖用户手填的类目）。
    """
    val = st.session_state.pop(pending_key, None)
    if val is None:
        return
    cur = str(st.session_state.get(widget_key, "") or "").strip()
    if overwrite or not cur:
        st.session_state[widget_key] = val


def reset_diagnosis(ns: str) -> None:
    """清空某 Tab 的诊断结果，保留数据与行业标准。"""
    for k in [f"{ns}_result", f"{ns}_batch_results", f"{ns}_failed_batches",
              f"{ns}_stop", f"{ns}_local_info", f"{ns}_n_batches"]:
        st.session_state.pop(k, None)


# ==========================================================
# 3. 行业基准展示（PRD §3.4.4 —— 必须区分实证与推断）
# ==========================================================

def _is_evidence(basis: str) -> bool:
    b = str(basis or "")
    return ("实证" in b) or ("检索" in b)


def render_industry_standard(std: IndustryStandard, ns: str, title: str = "行业标准") -> None:
    if std is None or (not std.ok and not std.raw_text):
        st.warning("尚未进行行业基准调研。请点击上方「🔍 AI 调研行业标准」。")
        return

    st.markdown(f"#### 📐 {title}")

    # --- 类目 / 市场 ---
    c1, c2, c3 = st.columns([2, 2, 1.2])
    with c1:
        st.markdown(f"**类目**：{std.category or '（未识别）'}")
    with c2:
        st.markdown(f"**目标市场**：{std.market or '（未识别）'}")
    with c3:
        conf = std.overall_confidence or "未提供"
        st.markdown(f"**整体置信度**：{conf}")

    # --- 未成功解析 ---
    if not std.ok:
        st.error(
            f"⚠️ AI 返回无法解析为结构化基准（{std.degrade_reason or '未知原因'}）。"
            "以下判定将降级为「店铺内部相对对比」，非行业标准。"
        )
        with st.expander("查看 AI 返回原文", expanded=False):
            st.code((std.raw_text or "")[:3000])
        return

    if std.salvaged:
        st.warning("⚠️ AI 返回的 JSON 存在语法错误，以下指标为抢救出的部分结果，可能不完整。")

    metrics = [m for m in (std.metrics or []) if isinstance(m, dict)]
    ev = [m for m in metrics if _is_evidence(m.get("basis"))]
    est = [m for m in metrics if not _is_evidence(m.get("basis"))]

    def _metric_card(m: dict, grey: bool):
        name = m.get("name") or m.get("key") or "未命名指标"
        unit = m.get("unit") or ""
        val = m.get("value")
        rng = m.get("range") or []
        rng_txt = f"（合理区间 {rng[0]}~{rng[1]}{unit}）" if len(rng) == 2 else ""
        conf = m.get("confidence") or ""
        src = m.get("source") or ""
        body = f"**{name}**：`{val}{unit}` {rng_txt}｜置信度：{conf}"
        if src:
            src_md = src if src.startswith("http") else f"_{src}_"
            body += f"\n\n来源：{src_md}"
        if grey:
            st.markdown(
                f'<div style="background:#F5F5F5;border-left:4px solid #BBBBBB;'
                f'padding:10px 12px;border-radius:6px;color:#666;margin:6px 0;">'
                f'🤖 经验推断（无公开数据）<br>{body}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="background:#F0F7FF;border-left:4px solid #2E7DFF;'
                f'padding:10px 12px;border-radius:6px;margin:6px 0;">'
                f'✅ 检索实证<br>{body}</div>',
                unsafe_allow_html=True,
            )

    if ev:
        st.caption(f"✅ 检索实证（{len(ev)} 项）")
        for m in ev:
            _metric_card(m, grey=False)
    if est:
        st.caption(f"🤖 经验推断（{len(est)} 项，模型明确标注无公开数据）")
        for m in est:
            _metric_card(m, grey=True)

    # --- 判定阈值 ---
    st.markdown("**判定阈值**")
    thr = std.thresholds or {}
    if thr:
        shown = {k: v for k, v in thr.items() if k != "note"}
        if shown:
            st.code(json.dumps(shown, ensure_ascii=False, indent=2), language="json")
        note = std.threshold_note()
        if note:
            st.markdown(f"依据：_{note}_")
        st.caption("🟢 以上阈值来自 AI 建议")
    else:
        st.markdown(
            '> ⚠️ **AI 未给出 thresholds，已使用系统默认兜底比例 0.7**'
            '（即低于基准 70% 视为显著偏低）'
        )

    # --- 市场行情洞察 ---
    if std.market_notes:
        with st.expander(f"💡 市场行情洞察（{len(std.market_notes)} 条）", expanded=False):
            for n in std.market_notes:
                st.markdown(f"- {n}")

    # --- 联网透明度 ---
    cols = st.columns([1, 1, 3])
    with cols[0]:
        st.caption(f"本次联网检索：**{std.search_count}** 次")
    with cols[1]:
        st.caption(f"调研时间：{std.created_at}")
    if std.search_summary:
        st.caption(f"🗒️ AI 检索自述：{std.search_summary}")
    elif std.search_count == 0:
        # usage 未上报检索次数、且 AI 也没给出检索自述时，才提示可能没联网
        st.caption("⚠️ 本次未能确认 AI 是否触发联网检索，基准可能来自其自身经验或本店数据，请谨慎参考。")
    if std.source_urls:
        with st.expander(f"🔗 AI 提供的来源链接（{len(std.source_urls)}）", expanded=False):
            for u in std.source_urls:
                st.markdown(f"- {u}")

    st.info("⚠️ 本基准由 AI 联网检索与自主判断生成，仅供参考，请结合店铺实际经营阶段判断。")


# ==========================================================
# 4. 指标卡
# ==========================================================

def metric_cards(items: list[tuple[str, str, str | None]]) -> None:
    """items: [(label, value, help)]"""
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        with col:
            st.metric(label, value, help=help_text, border=True)


# ==========================================================
# 5. 表格高亮渲染（PRD §6.4 性能红线）
# ==========================================================

def _style_row(row, red_prefixes, gold_prefixes):
    tag = str(row.get(COL_TAG, "")) if COL_TAG in row.index else str(row.get(COL_TAG_PLAIN, ""))
    plain = strip_tag_plain(tag)
    if plain in red_prefixes:
        return [f"background-color: {HIGHLIGHT_RED}"] * len(row)
    if plain in gold_prefixes:
        return [f"background-color: {HIGHLIGHT_GOLD}"] * len(row)
    if plain == "表现良好":
        return [f"background-color: {HIGHLIGHT_GREEN}"] * len(row)
    return [""] * len(row)


def render_result_table(
    df: pd.DataFrame,
    red_plains: list[str],
    gold_plains: list[str],
    render_rows: int = DEFAULT_RENDER_ROWS,
    height: int = TABLE_HEIGHT,
) -> None:
    """带高亮的表格渲染；超过 MAX_STYLE_ROWS 自动关闭 Styler 以免卡死。"""
    n = len(df)
    view_n = min(int(render_rows), n) if render_rows else n
    view = df.head(view_n)

    if len(view) <= MAX_STYLE_ROWS:
        styler = view.style.apply(_style_row, axis=1,
                                  red_prefixes=set(red_plains), gold_prefixes=set(gold_plains))
        st.dataframe(styler, width="stretch", height=height, hide_index=True)
    else:
        st.warning(
            f"渲染行数 {len(view)} 超过 {MAX_STYLE_ROWS}，已关闭底色高亮以提升响应速度"
            "（导出的 Excel 仍带高亮）。"
        )
        st.dataframe(view, width="stretch", height=height, hide_index=True,
                     column_config={COL_TAG: st.column_config.TextColumn(width="medium")})
    st.caption(f"展示 {len(view)} / 共 {n} 行；**导出包含全部 {n} 行**。")


def render_rows_selector(key: str, n: int) -> int:
    opts = RENDER_ROWS_OPTIONS
    labels = ["100 行", "500 行", "2000 行", f"全部（{n} 行）"]
    choice = st.selectbox("渲染行数", options=opts, format_func=lambda v: labels[opts.index(v)],
                          key=key, help="仅影响页面渲染速度，不影响导出结果")
    return n if choice >= n else int(choice)


# ==========================================================
# 6. 标签汇总条 / 筛选
# ==========================================================

def render_tag_summary_bar(counts: dict, total: int, ai_covered: int, allowed: list[str]) -> None:
    parts = [f"**共 {total} 条**"]
    for t in allowed:
        parts.append(f"{t} {counts.get(t, 0)}")
    st.markdown("｜".join(parts) + f"｜🤖 AI 精修 **{ai_covered}** 条")


def tag_filter(key: str, allowed: list[str], default_all: bool = True):
    options = [t for t in allowed]
    default = options if default_all else options
    return st.multiselect("按诊断标签筛选", options=options, default=default, key=key)


def source_filter(key: str):
    from config import SOURCE_AI, SOURCE_LOCAL
    return st.multiselect("按标签来源筛选", options=[SOURCE_AI, SOURCE_LOCAL],
                          default=[SOURCE_AI, SOURCE_LOCAL], key=key)


# ==========================================================
# 7. 下载按钮
# ==========================================================

def download_report_button(df: pd.DataFrame, std_text: str, prefix: str,
                           red_plains: list[str], gold_plains: list[str],
                           label: str = "⬇️ 下载诊断报告（Excel）") -> None:
    with st.spinner("正在生成 Excel 报告…"):
        try:
            data = to_excel_bytes(df, std_text, red_plains, gold_plains)
            st.download_button(
                label=label,
                data=data,
                file_name=report_filename(prefix),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )
        except Exception as e:
            st.error(f"生成 Excel 失败：{type(e).__name__}: {e}")


# ==========================================================
# 8. 预计耗时 / 批数
# ==========================================================

def estimate_call_info(n_ai_rows: int, rows_per_call: int, concurrency: int) -> tuple[int, str]:
    n_batches = math.ceil(n_ai_rows / rows_per_call) if n_ai_rows else 0
    if not n_batches:
        return 0, "—"
    waves = math.ceil(n_batches / max(concurrency, 1))
    sec = waves * 12          # 每波按 8~15s 保守估算
    lo, hi = int(sec * 0.7), int(sec * 1.4)
    txt = f"约 {lo}~{hi} 秒" if hi < 100 else f"约 {lo // 60}′{lo % 60:02d}″~{hi // 60}′{hi % 60:02d}″"
    return n_batches, txt


def show_api_error(e: Exception, context: str = "") -> None:
    """统一错误展示，不抛裸 traceback（PRD §5 交互细则）。"""
    if isinstance(e, MissingAPIKeyError):
        st.error(
            "❌ 未检测到 API Key。请在项目根目录 `.streamlit/secrets.toml` 中配置：\n\n"
            "```toml\nDASHSCOPE_API_KEY = \"sk-...\"\n```\n"
            "（云端部署请在 Streamlit Cloud → Settings → Secrets 中配置）"
        )
        return
    st.error(f"❌ {'AI 调用失败' if isinstance(e, LLMError) else '发生错误'}"
             f"{'：' + context if context else ''}：{e}")
