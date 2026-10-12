"""模块一：商品诊断（PRD §3）。

流程：① AI 自主调研行业标准 → ② 上传商品 Excel → ③ AI 对比诊断 → ④ 高亮表格 + Excel 导出
"""
from __future__ import annotations

import json
import math

import pandas as pd
import numpy as np
import streamlit as st

from config import (
    SHEET_PRODUCT, DEFAULT_HEADER_ROW_PRODUCT, COLS_PRODUCT_REQUIRED,
    AI_TOP_K, AI_TOP_K_MIN, AI_TOP_K_MAX, AI_FULL_LIMIT, CONCURRENCY,
    MAX_DISPLAY_NAME_LEN, PRODUCT_TAGS,
    COL_TAG, COL_SOURCE, COL_REASON, COL_SUGGESTION,
    SOURCE_AI, SOURCE_LOCAL, SOURCE_LOCAL_AI_FAIL,
)
from prompts import (
    PRODUCT_INDUSTRY_PROMPT, PRODUCT_DIAGNOSE_PROMPT, SYSTEM_PRODUCT_INDUSTRY,
    SYSTEM_PRODUCT_DIAGNOSE, EXTRA_REQUIREMENT_SUFFIX, SHOP_CONTEXT, SUGGESTION_RULES, render,
)
# ⚠️ 本模块顶层禁止再定义名为 render 的函数，否则会遮蔽 prompts.render，
#    导致 render(TEMPLATE, key=...) 报 "unexpected keyword argument"。
#    Tab 入口函数名固定为 render_tab。
from llm import qwen_chat, LLMError, extract_text_sources
from utils.clean import fmt_pct, fmt_int, truncate_series
from utils.loader import (
    get_sheet_names, pick_missing_sheet, read_raw, check_required,
    prepare_product, summarize_product, product_rows_for_llm,
)
from utils.diagnose import (
    IndustryStandard, build_industry_standard, local_tag_product, select_ai_rows,
    product_priority_score, plan_rows_per_call, run_batches, merge_results,
    parse_json_array, count_tags, ai_covered_count, strip_tag_plain, calc_batches,
)
from modules.common import (
    sget, sset, set_pending, apply_pending, reset_diagnosis, identify_category,
    research_standard, render_industry_standard, metric_cards, render_result_table,
    render_rows_selector,
    render_tag_summary_bar, tag_filter, source_filter, download_report_button,
    estimate_call_info, show_api_error, build_shop_profile,
)

NS = "prod"
# ★ 判据已由「询盘」改为「商机」；「流量不精准」同样是问题，一并高亮
RED_PLAINS = ["高曝光低点击", "高点击低商机", "流量不精准"]
GOLD_PLAINS: list[str] = []

RESULT_COLS = [COL_TAG, COL_SOURCE, "产品ID", "产品名称", "搜索曝光次数", "搜索点击次数",
               "点击率", "商机数", "商机率", COL_REASON, COL_SUGGESTION]


# ==========================================================
# 对外入口
# ==========================================================

def render_tab() -> None:
    _section_industry()
    st.divider()
    _section_upload()
    st.divider()
    _section_diagnose()
    st.divider()
    _section_result()


# ==========================================================
# ① 行业标准区
# ==========================================================

def _section_industry() -> None:
    # ⚠️ 必须在 text_input 实例化之前执行，否则会触发
    # StreamlitWidgetAlreadyInstantiatedError
    apply_pending("prod_cat_pending", "prod_cat_input")

    st.markdown("### ① 行业标准（由 AI 联网自主调研，人不定任何数值）")

    cat = st.text_input(
        "类目（留空则由 AI 自动识别）",
        key="prod_cat_input",
        placeholder="例如：阿里国际站 > 建材 > 无障碍扶手",
        help="上传数据后 AI 会自动抽样商品标题识别类目；你也可以随时手动覆盖。",
    )
    extra = st.text_input(
        "补充调研要求（选填）",
        key="prod_extra_input",
        placeholder="例如：只看北美市场 / 区分批发与零售 / 重点看近一年数据",
    )

    msg = sget("prod_cat_identified_msg")
    if msg:
        st.success(msg)
        st.session_state.pop("prod_cat_identified_msg", None)

    # --- 自动识别类目（PRD §3.4.1）：上传后对每个新文件触发一次 ---
    _maybe_auto_identify()

    df_loaded = sget("prod_df")
    if df_loaded is not None and len(df_loaded):
        st.caption(f"ℹ️ 已加载 **{len(df_loaded)}** 条商品，本次调研会**同时参考本店实际数据**与联网检索"
                   "——冷门品类查不到公开基准时，将以本店分布作为相对锚点。")
    else:
        st.caption("ℹ️ 尚未上传数据，本次调研仅依据联网检索。**建议先上传数据再调研**，基准会更贴合本店。")

    std: IndustryStandard | None = sget("prod_std")
    sig = (str(cat).strip(), str(extra).strip())
    prev_sig = sget("prod_std_sig")
    if std is not None and prev_sig is not None and tuple(prev_sig) != sig:
        st.warning("⚠️ 类目或补充要求已变更，原基准作废，请重新点击调研。")

    label = "🔄 重新调研行业标准" if std is not None else "🔍 AI 调研行业标准"
    if st.button(label, key="btn_industry_product", type="primary"):
        _do_research(str(cat).strip(), str(extra).strip(), sig)

    render_industry_standard(sget("prod_std"), NS, title="商品行业标准")


def _maybe_auto_identify() -> None:
    """上传文件后自动抽样标题识别类目（PRD §3.4.1）。每个文件只跑一次。"""
    df = sget("prod_df")
    if df is None or df.empty or "产品名称" not in df.columns:
        return
    sig = sget("prod_file_sig")
    if sget("prod_cat_identified_for") == sig:
        return
    if str(st.session_state.get("prod_cat_input", "") or "").strip():
        sset("prod_cat_identified_for", sig)
        return
    with st.spinner("AI 正在从商品标题识别类目…"):
        try:
            titles = df["产品名称"].dropna().astype(str).tolist()
            res = identify_category(titles)
        except Exception as e:
            st.caption(f"类目自动识别跳过：{e}")
            sset("prod_cat_identified_for", sig)
            return
    if res.get("error"):
        sset("prod_cat_identified_msg", f"类目自动识别失败：{res['error']}")
    elif res.get("category"):
        # 不能直接写 prod_cat_input（控件本轮已实例化），先暂存，下一轮开头落盘
        set_pending("prod_cat_pending", res["category"])
        sset("prod_market_suggested", res.get("market", ""))
        sset("prod_cat_identified_msg",
             f"🤖 AI 识别类目：**{res['category']}**（目标市场：{res.get('market', '—')}），已自动填入，可手动修改")
    sset("prod_cat_identified_for", sig)
    st.rerun()


def _do_research(category: str, extra: str, sig: tuple) -> None:
    # ★ 用户反馈 4：把本店上传的真实数据一并交给 AI，冷门品类才查得到/校准得了基准
    profile = build_shop_profile("product", sget("prod_df"), sget("prod_summary", {}),
                                 sget("prod_meta", {}))
    prompt = render(PRODUCT_INDUSTRY_PROMPT,
                    category=category or "",
                    market=str(sget("prod_market_suggested") or ""),
                    shop_profile=profile,
                    shop_context=SHOP_CONTEXT)
    if extra:
        prompt += render(EXTRA_REQUIREMENT_SUFFIX, extra=extra)
    if not category:
        st.caption("未指定类目，AI 将按自身判断确定研究对象的类目与目标市场。")

    try:
        with st.spinner("AI 正在联网调研行业标准，约需 10~30 秒…（如返回异常会自动重试一次）"):
            std, r, err = research_standard(
                prompt, SYSTEM_PRODUCT_INDUSTRY,
                category_default=category,
                market_default=str(sget("prod_market_suggested") or ""),
            )
    except LLMError as e:
        show_api_error(e, "行业基准调研")
        return

    sset("prod_std", std)
    sset("prod_std_sig", list(sig))
    if std.ok:
        msg = (f"✅ 调研完成（本次联网检索 {r.search_count} 次，"
               f"指标 {len(std.metrics)} 项，阈值 {len(std.thresholds)} 项）")
        if std.salvaged:
            msg += "｜⚠️ AI 返回存在语法错误，已抢救出可用指标"
        st.success(msg)
    else:
        st.warning(f"⚠️ {err}。已保留原文，请重试或调整补充要求。")
    st.rerun()


# ==========================================================
# ② 数据上传区
# ==========================================================

def _section_upload() -> None:
    st.markdown("### ② 上传店铺商品数据")

    c1, c2 = st.columns([3, 1])
    with c1:
        up = st.file_uploader("拖拽或点击上传", type=["xls", "xlsx", "csv"],
                              key="prod_uploader", label_visibility="collapsed")
    with c2:
        header_row = st.number_input("表头所在行号", min_value=1, max_value=20,
                                     value=DEFAULT_HEADER_ROW_PRODUCT, key="prod_header_row",
                                     help="默认的 6 是按实测文件核对的：表头在第 6 行")

    if up is None:
        st.caption("支持 .xls / .xlsx / .csv。上传后自动读取 `产品分析` Sheet。")
        st.markdown(
            '<div style="color:#8A6D3B;background:#FCF8E3;border-left:4px solid #F0AD4E;'
            'padding:8px 12px;border-radius:6px;font-size:13px;">'
            '⚠️ 上传数据将用于 AI 分析，请勿上传含客户隐私信息的表单。</div>',
            unsafe_allow_html=True,
        )
        return

    file_sig = (up.name, up.size, int(header_row))
    if sget("prod_file_sig") != file_sig:
        _load_file(up, int(header_row), file_sig)
        st.rerun()

    df = sget("prod_df")
    meta = sget("prod_meta", {})
    if df is None:
        return

    st.success(f"✅ 已加载 **{len(df)}** 条商品")
    dropped = meta.get("rows_dropped_no_id", 0) + meta.get("rows_dropped_dup", 0)
    if dropped:
        st.caption(f"（读取 {meta.get('rows_raw', 0)} 行，"
                   f"丢弃无 ID {meta.get('rows_dropped_no_id', 0)} 行、"
                   f"重复 ID {meta.get('rows_dropped_dup', 0)} 行）")
    if meta.get("zero_exposure"):
        st.caption(f"其中 {meta['zero_exposure']} 条商品曝光为 0，判定为「样本不足」，不参与异常标签判定。")

    s = sget("prod_summary", {})
    metric_cards([
        ("商品总数", fmt_int(s.get("n_total", 0)), None),
        ("总曝光", fmt_int(s.get("impressions", 0)), None),
        ("总点击", fmt_int(s.get("clicks", 0)), None),
        ("总商机", fmt_int(s.get("opportunities", 0)),
         "商机数 = 询盘个数 + TM咨询人数"),
        ("全店加权点击率", fmt_pct(s.get("ctr_all")), "总点击 / 总曝光"),
        ("全店加权商机率", fmt_pct(s.get("opp_all")), "总商机 / 总点击"),
        ("零商机商品占比", fmt_pct(s.get("zero_opp_ratio")), None),
    ])
    st.caption("口径：点击率 = 搜索点击次数 / 搜索曝光次数；"
               "**商机数 = 询盘个数 + TM咨询人数**，商机率 = 商机数 / 搜索点击次数。"
               "（买家多数走 TM 咨询而非表单询盘，故以商机为核心判据）")

    with st.expander(f"数据预览（前 10 行 / 共 {len(df)} 行）", expanded=False):
        preview = df[["产品ID", "产品名称", "搜索曝光次数", "搜索点击次数",
                      "点击率_计算", "商机数", "商机率_计算"]].head(10).copy()
        preview["产品名称"] = truncate_series(preview["产品名称"], MAX_DISPLAY_NAME_LEN)
        preview["点击率_计算"] = preview["点击率_计算"].map(fmt_pct)
        preview["商机率_计算"] = preview["商机率_计算"].map(fmt_pct)
        st.dataframe(preview, width="stretch", hide_index=True)

    with st.expander("⚙️ 高级设置", expanded=False):
        a1, a2 = st.columns(2)
        with a1:
            st.number_input("AI 精修上限", min_value=AI_TOP_K_MIN, max_value=AI_TOP_K_MAX,
                            value=AI_TOP_K, step=50, key="prod_top_k",
                            help=f"商品数 > {AI_FULL_LIMIT} 时，AI 只对权重最高的前 K 条精修，"
                                 f"其余由本地规则按 AI 基准判定")
        with a2:
            st.slider("并发批数", min_value=1, max_value=8, value=CONCURRENCY,
                      key="prod_conc", help="DashScope 有 QPS 限制，过高会触发 429")


def _load_file(up, header_row: int, file_sig: tuple) -> None:
    try:
        sheet, fallback, sheets = pick_missing_sheet(up, SHEET_PRODUCT)
    except Exception as e:
        st.error(f"无法读取该文件：{type(e).__name__}: {e}")
        for k in ["prod_df", "prod_meta", "prod_summary", "prod_file_sig"]:
            st.session_state.pop(k, None)
        return

    if fallback:
        st.warning(f"未找到 Sheet `{SHEET_PRODUCT}`，已回退到 `{sheet}`。")
        if sheets:
            chosen = st.selectbox("请选择正确的 Sheet", options=sheets,
                                  index=0, key="prod_sheet_select")
            sheet = chosen

    try:
        raw = read_raw(up, sheet, header_row)
    except Exception as e:
        st.error(f"读取 Sheet `{sheet}` 失败：{type(e).__name__}: {e}")
        return

    missing = check_required(raw, COLS_PRODUCT_REQUIRED)
    if missing:
        st.error(
            f"❌ 第 {header_row} 行未解析到必需列，缺失：`{'`、`'.join(missing)}`。\n\n"
            f"已识别到的列名：{list(raw.columns)[:15]}...\n\n"
            f"请调整「表头所在行号」后重试。"
        )
        for k in ["prod_df", "prod_meta", "prod_summary", "prod_file_sig"]:
            st.session_state.pop(k, None)
        return

    df, meta = prepare_product(raw)
    if df.empty:
        st.error("解析后无有效数据行（可能所有行的产品ID都为空）。")
        return

    sset("prod_df", df)
    sset("prod_meta", meta)
    sset("prod_summary", summarize_product(df))
    sset("prod_file_sig", file_sig)
    reset_diagnosis(NS)
    st.session_state.pop("prod_cat_identified_for", None)


# ==========================================================
# ③ 诊断区
# ==========================================================

def _section_diagnose() -> None:
    st.markdown("### ③ 开始诊断")

    df = sget("prod_df")
    std = sget("prod_std")
    ready = df is not None and not df.empty and std is not None and getattr(std, "ok", False)
    if not ready:
        hint = []
        if df is None or df.empty:
            hint.append("上传商品数据")
        if std is None or not getattr(std, "ok", False):
            hint.append("完成行业基准调研")
        st.button("🚀 开始诊断", disabled=True, key="btn_run_product")
        st.caption(f"请先完成：{' 与 '.join(hint)}。")
        return

    n = len(df)
    top_k = int(sget("prod_top_k") or AI_TOP_K)
    rows_per_call = plan_rows_per_call(df["产品名称"])
    idx_rows, mode = select_ai_rows(df, None, top_k, AI_FULL_LIMIT)
    n_batches, eta = estimate_call_info(len(idx_rows), rows_per_call, int(sget("prod_conc") or CONCURRENCY))

    if mode == "hybrid":
        st.info(
            f"ℹ️ 当前共 {n} 个商品，超过全量上限 {AI_FULL_LIMIT}，"
            f"AI 将精修权重最高的 {len(idx_rows)} 个高价值商品，"
            f"其余 {n - len(idx_rows)} 个由本地规则依据 AI 基准自动判定"
            f"（`标签来源` 列标明「本地规则」）。如需 AI 全量精修，请调高上方「AI 精修上限」。"
        )
    st.caption(f"本次将调用 ceil({len(idx_rows)}/{rows_per_call}) = **{n_batches}** 次 API，"
               f"预计耗时 **{eta}**（并发 {int(sget('prod_conc') or CONCURRENCY)}）")
    if n_batches > 3:
        st.caption("💡 首次使用建议把上方「AI 精修上限」调小（如 60），"
                   "先跑 1~2 批看建议质量，满意后再调大跑全量——"
                   "否则整轮会持续几分钟，期间页面在等 AI 返回。")

    running = sget("prod_running", False)
    b1, b2, b3 = st.columns([1.2, 1, 2])

    with b1:
        if st.button("🚀 开始诊断" if not running else "⏳ 诊断中…",
                     key="btn_run_product", type="primary", disabled=running):
            _start_diagnosis(idx_rows, rows_per_call, mode)
            st.rerun()
    with b2:
        if running:
            if st.button("⏹ 中止", key="btn_stop_product"):
                sset("prod_stop", True)
                st.rerun()
    with b3:
        failed = list(sget("prod_failed_batches", []) or [])
        if failed and not running:
            if st.button(f"🔁 仅重试失败批次（{len(failed)}）", key="btn_retry_product"):
                st.session_state["prod_attempts"] = {}
                sset("prod_failed_batches", [])
                sset("prod_stop", False)
                sset("prod_running", True)
                st.rerun()

    if running:
        _exec_wave()


def _start_diagnosis(idx_rows: list, rows_per_call: int, mode: str) -> None:
    df = sget("prod_df").copy()
    std = sget("prod_std")
    summary = sget("prod_summary", {})

    # --- 本地规则预筛（PRD §3.5.1）：阈值取自 AI ---
    res = local_tag_product(df, std, summary)
    df["_local_tag"] = res["series"].values
    sset("prod_local_info", res["info"])

    # --- Top-K 排序（混合模式）---
    score = product_priority_score(df, df["_local_tag"])
    idx_rows2, mode2 = select_ai_rows(df, score, int(sget("prod_top_k") or AI_TOP_K), AI_FULL_LIMIT)

    sset("prod_df", df)
    sset("prod_plan", {
        "idx_rows": idx_rows2, "rows_per_call": rows_per_call, "mode": mode2,
        "conc": int(sget("prod_conc") or CONCURRENCY),
    })
    sset("prod_batch_results", {})
    sset("prod_failed_batches", [])
    sset("prod_attempts", {})
    sset("prod_stop", False)
    sset("prod_finished", False)
    sset("prod_running", True)
    st.session_state.pop("prod_result", None)


def _exec_wave() -> None:
    plan = sget("prod_plan") or {}
    df = sget("prod_df")
    std = sget("prod_std")
    summary = sget("prod_summary", {})
    if df is None or std is None or not plan:
        sset("prod_running", False)
        return

    idx_rows = plan.get("idx_rows", [])
    rows_per_call = plan.get("rows_per_call", 50)
    n_ai = len(idx_rows)

    prog = st.progress(0.0, text="准备中…")
    status_ph = st.empty()

    std_json = json.dumps({
        "category": std.category, "market": std.market,
        "metrics": std.metrics, "thresholds": std.thresholds,
        "market_notes": std.market_notes,
    }, ensure_ascii=False)

    def task(chunk: pd.DataFrame, bidx: int):
        rows = product_rows_for_llm(chunk)
        prompt = render(
            PRODUCT_DIAGNOSE_PROMPT,
            industry_standard_json=std_json,
            shop_context=SHOP_CONTEXT,
            suggestion_rules=SUGGESTION_RULES,
            n_total=summary.get("n_total", 0),
            ctr_all=fmt_pct(summary.get("ctr_all")),
            opp_all=fmt_pct(summary.get("opp_all")),
            opp_total=int(summary.get("opportunities", 0)),
            zero_opp_ratio=fmt_pct(summary.get("zero_opp_ratio")),
            exp_p50=int(summary.get("exp_p50", 0)),
            clk_p50=int(summary.get("clk_p50", 0)),
            rows_json=json.dumps(rows, ensure_ascii=False),
        )
        try:
            r = qwen_chat(
                messages=[{"role": "system", "content": SYSTEM_PRODUCT_DIAGNOSE},
                          {"role": "user", "content": prompt}],
                enable_search=False, temperature=0.1, max_tokens=6000,
            )
        except LLMError as e:
            return False, str(e)
        arr = parse_json_array(r.text)
        if not arr:
            return False, "AI 返回无法解析为 JSON 数组"
        return True, arr

    def cb(done, total, nfail):
        rows_done = min(done * rows_per_call, n_ai)
        prog.progress(min(done / max(total, 1), 1.0),
                      text=f"已完成 {done}/{total} 批（{rows_done}/{n_ai} 行）")
        if nfail:
            status_ph.warning(f"{nfail} 批异常，稍后可点击「仅重试失败批次」")

    run_batches(idx_rows, df, task, NS,
                max_rows=rows_per_call, concurrency=plan.get("conc", CONCURRENCY),
                progress_cb=cb, max_waves=1)

    if sget("prod_stop"):
        prog.empty()
        status_ph.info("⏹ 已中止，已完成批次结果保留。")
        sset("prod_running", False)
        _finalize(partial=True)
        st.rerun()

    if sget("prod_finished"):
        prog.progress(1.0, text="全部批次完成")
        sset("prod_running", False)
        _finalize(partial=False)
        st.rerun()

    st.rerun()


def _finalize(partial: bool = False) -> None:
    df = sget("prod_df")
    results = sget("prod_batch_results", {}) or {}
    failed = sget("prod_failed_batches", []) or []
    info = sget("prod_local_info", {})

    if partial and not results:
        return

    out = merge_results(df, results, id_col="产品ID", allowed=PRODUCT_TAGS)
    # 失败批次覆盖的行标注来源
    sset("prod_result", out)
    sset("prod_running", False)
    if not partial:
        st.session_state["prod_last_run"] = {
            "n": len(out), "failed": len(failed),
            "mode": (sget("prod_plan") or {}).get("mode", ""),
        }


# ==========================================================
# ④ 结果区
# ==========================================================

def _section_result() -> None:
    st.markdown("### ④ 诊断结果")
    out = sget("prod_result")
    if out is None:
        st.caption("尚未生成诊断结果。完成上方三步后自动显示。")
        return

    info = sget("prod_local_info", {})
    if info.get("degraded"):
        st.error("⚠️ 行业基准不可用，以下判定已降级为「店铺内部相对对比」，非行业标准。")
    if not info.get("thr_source", {}).get("ctr", "").startswith("AI"):
        st.warning('⚠️ AI 未给出 ctr 阈值，已使用系统默认 0.7 判定。')
    if not info.get("thr_source", {}).get("opportunity_rate", "").startswith("AI"):
        st.warning('⚠️ AI 未给出 opportunity_rate（商机率）阈值，已使用系统默认 0.7 判定。')
    for note in info.get("notes", []):
        st.caption(f"ℹ️ {note}")

    total = len(out)
    counts = count_tags(out, PRODUCT_TAGS)
    render_tag_summary_bar(counts, total, ai_covered_count(out), PRODUCT_TAGS)

    f1, f2, f3 = st.columns([2, 1.4, 1])
    with f1:
        tags = tag_filter("prod_tag_filter", PRODUCT_TAGS)
    with f2:
        sources = source_filter("prod_source_filter")
    with f3:
        imp_min = st.slider("曝光量下限", 0, int(out["搜索曝光次数"].max() or 1), 0,
                            key="prod_imp_min")

    view = out.copy()
    if tags:
        view = view[view[COL_TAG].isin(tags)]
    if sources:
        view = view[view[COL_SOURCE].isin(sources)]
    if imp_min:
        view = view[view["搜索曝光次数"] >= imp_min]

    if view.empty:
        st.info("当前筛选条件下没有数据。")
        return

    disp = view.copy()
    if "产品名称" in disp.columns:
        disp["产品名称"] = truncate_series(disp["产品名称"], MAX_DISPLAY_NAME_LEN)
    disp["点击率"] = disp["点击率_计算"].map(fmt_pct)
    disp["商机率"] = disp["商机率_计算"].map(fmt_pct)
    cols = [c for c in RESULT_COLS if c in disp.columns]
    disp = disp[cols]

    render_rows = render_rows_selector("prod_render_rows", len(disp))
    render_result_table(disp, RED_PLAINS, GOLD_PLAINS, render_rows=render_rows)

    std_text = (sget("prod_std") or IndustryStandard()).to_text()
    download_report_button(out[[c for c in RESULT_COLS if c in out.columns]],
                           std_text, "商品诊断报告", RED_PLAINS, GOLD_PLAINS,
                           label=f"⬇️ 下载商品诊断报告（Excel，全量 {len(out)} 行）")
