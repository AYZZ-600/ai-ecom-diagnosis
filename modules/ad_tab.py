"""模块二：广告诊断（PRD §4）。

与模块一共享骨架，差异在于 Sheet 名、字段、清洗规则、标签体系。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    SHEET_AD, DEFAULT_HEADER_ROW_AD, COLS_AD_REQUIRED,
    AI_TOP_K, AI_TOP_K_MIN, AI_TOP_K_MAX, AI_FULL_LIMIT, CONCURRENCY,
    AD_MIN_COST_FOR_JUDGE, AD_MIN_CLICKS_FOR_JUDGE,
    MAX_DISPLAY_NAME_LEN, AD_TAGS, AD_TAG_BURN, AD_TAG_GOLD, AD_TAG_TRAFFIC,
    AD_TAG_USELESS, AD_TAG_NORMAL,
    COL_TAG, COL_SOURCE, COL_REASON, COL_SUGGESTION,
)
from prompts import (
    AD_INDUSTRY_PROMPT, AD_DIAGNOSE_PROMPT, SYSTEM_AD_INDUSTRY,
    SYSTEM_AD_DIAGNOSE, EXTRA_REQUIREMENT_SUFFIX, SHOP_CONTEXT, SUGGESTION_RULES,
    AD_SUGGESTION_EXTRA, render,
)
# ⚠️ 本模块顶层禁止再定义名为 render 的函数（会遮蔽 prompts.render）。
#    Tab 入口函数名固定为 render_tab。
from llm import qwen_chat, LLMError, extract_text_sources
from utils.clean import fmt_pct, fmt_money, fmt_int, truncate_series
from utils.loader import (
    pick_missing_sheet, read_raw, check_required,
    prepare_ad, summarize_ad, ad_rows_for_llm,
)
from utils.diagnose import (
    IndustryStandard, build_industry_standard, local_tag_ad, select_ai_rows,
    plan_rows_per_call, run_batches, merge_results, parse_json_array,
    count_tags, ai_covered_count, calc_batches, strip_forbidden_clauses,
)
from modules.common import (
    sget, sset, set_pending, apply_pending, reset_diagnosis, identify_category,
    research_standard, render_industry_standard, metric_cards, render_result_table,
    render_rows_selector,
    render_tag_summary_bar, tag_filter, source_filter, download_report_button,
    estimate_call_info, show_api_error, build_shop_profile,
)

NS = "ad"
# PRD §4.6：🔴 浅红；🟡 浅金；并沿用 §6.4 把「⚪ 鸡肋品」也按异常处理
RED_PLAINS = ["烧钱品", "流量品", "鸡肋品"]
GOLD_PLAINS = ["黄金品"]

RESULT_COLS = [COL_TAG, COL_SOURCE, "产品ID", "商品信息", "花费", "商机数", "商机成本",
               "点击量", "曝光量", "点击率", "商机转化率", COL_REASON, COL_SUGGESTION]


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
# ① 广告行业标准区
# ==========================================================

def _section_industry() -> None:
    # ⚠️ 必须在 text_input 实例化之前执行，否则会触发
    # StreamlitWidgetAlreadyInstantiatedError
    apply_pending("ad_cat_pending", "ad_cat_input")

    st.markdown("### ① 广告投放标准（由 AI 联网自主调研，人不定任何数值）")

    cat = st.text_input(
        "类目（留空则由 AI 自动识别）",
        key="ad_cat_input",
        placeholder="例如：阿里国际站 > 建材 > 无障碍扶手",
    )
    extra = st.text_input(
        "补充调研要求（选填）",
        key="ad_extra_input",
        placeholder="例如：只看北美市场 / 重点看近半年 / 区分关键词热度层级",
    )

    msg = sget("ad_cat_identified_msg")
    if msg:
        st.success(msg)
        st.session_state.pop("ad_cat_identified_msg", None)

    _maybe_auto_identify()

    df_loaded = sget("ad_df")
    if df_loaded is not None and len(df_loaded):
        st.caption(f"ℹ️ 已加载 **{len(df_loaded)}** 条推广数据，本次调研会**同时参考本店实际投放数据**"
                   "与联网检索（本店投放样本小，纯公开基准往往不适用）。")
    else:
        st.caption("ℹ️ 尚未上传数据，本次调研仅依据联网检索。**建议先上传数据再调研**，基准会更贴合本店。")

    std = sget("ad_std")
    sig = (str(cat).strip(), str(extra).strip())
    prev_sig = sget("ad_std_sig")
    if std is not None and prev_sig is not None and tuple(prev_sig) != sig:
        st.warning("⚠️ 类目或补充要求已变更，原基准作废，请重新点击调研。")

    label = "🔄 重新调研广告行业标准" if std is not None else "🔍 AI 调研广告行业标准"
    if st.button(label, key="btn_industry_ad", type="primary"):
        _do_research(str(cat).strip(), str(extra).strip(), sig)

    render_industry_standard(sget("ad_std"), NS, title="广告投放行业标准")


def _maybe_auto_identify() -> None:
    df = sget("ad_df")
    if df is None or df.empty or "商品信息" not in df.columns:
        return
    sig = sget("ad_file_sig")
    if sget("ad_cat_identified_for") == sig:
        return
    if str(st.session_state.get("ad_cat_input", "") or "").strip():
        sset("ad_cat_identified_for", sig)
        return
    with st.spinner("AI 正在从商品标题识别类目…"):
        try:
            titles = df["商品信息"].dropna().astype(str).tolist()
            res = identify_category(titles)
        except Exception as e:
            st.caption(f"类目自动识别跳过：{e}")
            sset("ad_cat_identified_for", sig)
            return
    if res.get("category"):
        # 不能直接写 ad_cat_input（控件本轮已实例化），先暂存，下一轮开头落盘
        set_pending("ad_cat_pending", res["category"])
        sset("ad_market_suggested", res.get("market", ""))
        sset("ad_cat_identified_msg",
             f"🤖 AI 识别类目：**{res['category']}**（目标市场：{res.get('market', '—')}），已自动填入，可手动修改")
    sset("ad_cat_identified_for", sig)
    st.rerun()


def _do_research(category: str, extra: str, sig: tuple) -> None:
    # ★ 用户反馈 4：把本店实际投放数据一并交给 AI（本店投放样本很小，纯联网基准不适用）
    profile = build_shop_profile("ad", sget("ad_df"), sget("ad_summary", {}), sget("ad_meta", {}))
    prompt = render(AD_INDUSTRY_PROMPT,
                    category=category or "",
                    market=str(sget("ad_market_suggested") or ""),
                    shop_profile=profile,
                    shop_context=SHOP_CONTEXT)
    if extra:
        prompt += render(EXTRA_REQUIREMENT_SUFFIX, extra=extra)

    try:
        with st.spinner("AI 正在联网调研广告投放基准，约需 10~30 秒…（如返回异常会自动重试一次）"):
            std, r, err = research_standard(
                prompt, SYSTEM_AD_INDUSTRY,
                category_default=category,
                market_default=str(sget("ad_market_suggested") or ""),
            )
    except LLMError as e:
        show_api_error(e, "广告基准调研")
        return

    sset("ad_std", std)
    sset("ad_std_sig", list(sig))
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
    st.markdown("### ② 上传推广商品广告数据")

    c1, c2 = st.columns([3, 1])
    with c1:
        up = st.file_uploader("拖拽或点击上传", type=["xls", "xlsx", "csv"],
                              key="ad_uploader", label_visibility="collapsed")
    with c2:
        header_row = st.number_input("表头所在行号", min_value=1, max_value=20,
                                     value=DEFAULT_HEADER_ROW_AD, key="ad_header_row",
                                     help="广告文件默认表头在第 1 行")

    if up is None:
        st.caption("支持 .xls / .xlsx / .csv。上传后自动读取 `sheet1`。")
        return

    file_sig = (up.name, up.size, int(header_row))
    if sget("ad_file_sig") != file_sig:
        _load_file(up, int(header_row), file_sig)
        st.rerun()

    df = sget("ad_df")
    meta = sget("ad_meta", {})
    if df is None:
        return

    st.success(f"✅ 已加载 **{len(df)}** 条推广商品"
               f"（其中 **{meta.get('rows_invested', 0)}** 条有实际投放花费）")

    s = sget("ad_summary", {})
    metric_cards([
        ("推广品数", fmt_int(s.get("n", 0)), None),
        ("总花费", fmt_money(s.get("total_cost")), None),
        ("总商机", fmt_int(s.get("total_leads", 0)), "商机数 = 商机量 + TM咨询量"),
        ("总点击", fmt_int(s.get("total_clicks", 0)), None),
        ("整体 CPC", fmt_money(s.get("cpc_all")), "总花费 / 总点击"),
        ("整体商机转化率", fmt_pct(s.get("lead_rate_all")), "总商机数 / 总点击"),
        ("零商机花费占比", fmt_pct(s.get("zero_lead_cost_ratio")),
         "花费>0 但商机=0 的花费 / 总花费"),
    ])
    st.caption("口径：**商机数 = 商机量 + TM咨询量**（与商品模块一致，计入 TM 咨询转化）")

    if meta.get("rows_invested", 0) == 0:
        st.warning("⚠️ 本次数据无任何实际投放花费，无需广告诊断。")

    with st.expander(f"数据预览（前 10 行 / 共 {len(df)} 行）", expanded=False):
        preview = df[["产品ID", "商品信息", "花费", "商机数", "点击量", "曝光量",
                      "CTR_计算", "是否投放"]].head(10).copy()
        preview["商品信息"] = truncate_series(preview["商品信息"], MAX_DISPLAY_NAME_LEN)
        preview["花费"] = preview["花费"].map(fmt_money)
        preview["CTR_计算"] = preview["CTR_计算"].map(fmt_pct)
        st.dataframe(preview, width="stretch", hide_index=True)

    with st.expander("⚙️ 高级设置", expanded=False):
        a1, a2 = st.columns(2)
        with a1:
            st.number_input("AI 精修上限", min_value=AI_TOP_K_MIN, max_value=AI_TOP_K_MAX,
                            value=AI_TOP_K, step=50, key="ad_top_k")
        with a2:
            st.slider("并发批数", min_value=1, max_value=8, value=CONCURRENCY, key="ad_conc")


def _load_file(up, header_row: int, file_sig: tuple) -> None:
    try:
        sheet, fallback, sheets = pick_missing_sheet(up, SHEET_AD)
    except Exception as e:
        st.error(f"无法读取该文件：{type(e).__name__}: {e}")
        for k in ["ad_df", "ad_meta", "ad_summary", "ad_file_sig"]:
            st.session_state.pop(k, None)
        return

    if fallback:
        st.warning(f"未找到 Sheet `{SHEET_AD}`，已回退到 `{sheet}`。")
        if sheets:
            sheet = st.selectbox("请选择正确的 Sheet", options=sheets,
                                 index=0, key="ad_sheet_select")

    try:
        raw = read_raw(up, sheet, header_row)
    except Exception as e:
        st.error(f"读取 Sheet `{sheet}` 失败：{type(e).__name__}: {e}")
        return

    missing = check_required(raw, COLS_AD_REQUIRED)
    if missing:
        st.error(
            f"❌ 第 {header_row} 行未解析到必需列，缺失：`{'`、`'.join(missing)}`。\n\n"
            f"已识别到的列名：{list(raw.columns)[:15]}...\n\n"
            f"请调整「表头所在行号」后重试。"
        )
        for k in ["ad_df", "ad_meta", "ad_summary", "ad_file_sig"]:
            st.session_state.pop(k, None)
        return

    df, meta = prepare_ad(raw)
    if df.empty:
        st.error("解析后无有效数据行。")
        return

    sset("ad_df", df)
    sset("ad_meta", meta)
    sset("ad_summary", summarize_ad(df))
    sset("ad_file_sig", file_sig)
    reset_diagnosis(NS)
    st.session_state.pop("ad_cat_identified_for", None)


# ==========================================================
# ③ 诊断区
# ==========================================================

def _section_diagnose() -> None:
    st.markdown("### ③ 开始诊断")

    df = sget("ad_df")
    std = sget("ad_std")
    ready = df is not None and not df.empty and std is not None and getattr(std, "ok", False)
    if not ready:
        hint = []
        if df is None or df.empty:
            hint.append("上传广告数据")
        if std is None or not getattr(std, "ok", False):
            hint.append("完成广告基准调研")
        st.button("🚀 开始诊断", disabled=True, key="btn_run_ad")
        st.caption(f"请先完成：{' 与 '.join(hint)}。")
        return

    n = len(df)
    invested_n = int((df["是否投放"]).sum())
    if invested_n == 0:
        st.warning("本次数据无任何实际投放花费，全部商品将被标记为「⚫ 未投放（待激活）」，"
                   "无需调用 AI 做广告诊断。")
        if st.button("生成诊断结果（全部标记为未投放）", key="btn_run_ad_local"):
            _finalize_without_ai()
            st.rerun()
        return

    local = local_tag_ad(df, std, sget("ad_summary", {}))
    rows_per_call = plan_rows_per_call(df["商品信息"])
    idx_rows, mode = select_ai_rows(df, None, int(sget("ad_top_k") or AI_TOP_K), AI_FULL_LIMIT)
    n_batches, eta = estimate_call_info(len(idx_rows), rows_per_call, int(sget("ad_conc") or CONCURRENCY))

    st.caption(f"本地规则模式：{local['info'].get('mode', '')}")
    st.caption(f"判定门槛：花费 = 0 → ⚫ 未投放（待激活）；"
               f"花费 < {AD_MIN_COST_FOR_JUDGE:g} 元 且 点击 < {AD_MIN_CLICKS_FOR_JUDGE} 次 "
               f"→ 🧪 测试中（样本不足），不判烧钱品")
    if mode == "hybrid":
        st.info(f"ℹ️ 共 {n} 条推广商品，AI 精修花费最高的 {len(idx_rows)} 条，"
                f"其余由本地规则依据 AI 基准判定。")
    st.caption(f"本次将调用 **{n_batches}** 次 API，预计耗时 **{eta}**")

    running = sget("ad_running", False)
    b1, b2, b3 = st.columns([1.2, 1, 2])
    with b1:
        if st.button("🚀 开始诊断" if not running else "⏳ 诊断中…",
                     key="btn_run_ad", type="primary", disabled=running):
            _start_diagnosis(rows_per_call, mode)
            st.rerun()
    with b2:
        if running and st.button("⏹ 中止", key="btn_stop_ad"):
            sset("ad_stop", True)
            st.rerun()
    with b3:
        failed = sget("ad_failed_batches", []) or []
        if failed and not running:
            if st.button(f"🔁 仅重试失败批次（{len(failed)}）", key="btn_retry_ad"):
                st.session_state["ad_attempts"] = {}
                sset("ad_failed_batches", [])
                sset("ad_stop", False)
                sset("ad_running", True)
                st.rerun()

    if running:
        _exec_wave()


def _ad_score(df: pd.DataFrame) -> pd.Series:
    """Top-K 权重：优先「花费降序」（PRD §4.5.2），并保证已投放行先于未投放行。"""
    invested = df["是否投放"]
    return np.where(invested, 1e9, 0.0) + df["花费"].astype(float)


def _start_diagnosis(rows_per_call: int, mode: str) -> None:
    df = sget("ad_df").copy()
    std = sget("ad_std")
    summary = sget("ad_summary", {})

    res = local_tag_ad(df, std, summary)
    df["_local_tag"] = res["series"].values
    sset("ad_local_info", res["info"])

    if res["info"].get("mode") == "M=0":
        _finalize_without_ai()
        return

    score = pd.Series(_ad_score(df), index=df.index)
    idx_rows, mode2 = select_ai_rows(df, score, int(sget("ad_top_k") or AI_TOP_K), AI_FULL_LIMIT)

    sset("ad_df", df)
    sset("ad_plan", {"idx_rows": idx_rows, "rows_per_call": rows_per_call,
                     "mode": mode2, "conc": int(sget("ad_conc") or CONCURRENCY)})
    sset("ad_batch_results", {})
    sset("ad_failed_batches", [])
    sset("ad_attempts", {})
    sset("ad_stop", False)
    sset("ad_finished", False)
    sset("ad_running", True)
    st.session_state.pop("ad_result", None)


def _finalize_without_ai() -> None:
    df = sget("ad_df").copy()
    std = sget("ad_std")
    summary = sget("ad_summary", {})
    res = local_tag_ad(df, std, summary)
    df["_local_tag"] = res["series"].values
    sset("ad_local_info", res["info"])
    sset("ad_df", df)
    sset("ad_result", merge_results(df, {}, id_col="产品ID", allowed=AD_TAGS))
    sset("ad_running", False)
    sset("ad_finished", True)


def _exec_wave() -> None:
    plan = sget("ad_plan") or {}
    df = sget("ad_df")
    std = sget("ad_std")
    summary = sget("ad_summary", {})
    if df is None or std is None or not plan:
        sset("ad_running", False)
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
        # 传入 std：逐行计算 ctr_status，让 AI 明确知道这行是"流量端"还是"转化端"问题
        rows = ad_rows_for_llm(chunk, std)
        prompt = render(
            AD_DIAGNOSE_PROMPT,
            ad_industry_standard_json=std_json,
            shop_context=SHOP_CONTEXT,
            suggestion_rules=SUGGESTION_RULES,
            ad_suggestion_extra=AD_SUGGESTION_EXTRA,
            n=summary.get("n", 0),
            n_invested=summary.get("n_invested", 0),
            cost=float(summary.get("total_cost", 0)),
            leads=int(summary.get("total_leads", 0)),
            cpc=0 if not np.isfinite(summary.get("cpc_all", float("nan")))
            else round(float(summary["cpc_all"]), 2),
            lead_rate=fmt_pct(summary.get("lead_rate_all")),
            rows_json=json.dumps(rows, ensure_ascii=False),
        )
        try:
            r = qwen_chat(
                messages=[{"role": "system", "content": SYSTEM_AD_DIAGNOSE},
                          {"role": "user", "content": prompt}],
                enable_search=False, temperature=0.1, max_tokens=6000,
            )
        except LLMError as e:
            return False, str(e)
        arr = parse_json_array(r.text)
        if not arr:
            return False, "AI 返回无法解析为 JSON 数组"
        # ★ 确定性兜底：ctr_status=above（点击率达标）的行，
        #   说明流量端有效，剥离掉"改主图/改标题"这类方向错误的分句
        status_map = {str(x.get("product_id")): x.get("ctr_status") for x in rows}
        for item in arr:
            if status_map.get(str(item.get("product_id"))) == "above" and item.get("suggestion"):
                item["suggestion"] = strip_forbidden_clauses(
                    item["suggestion"],
                    fallback="详情页补齐认证与参数表、明确 MOQ 与阶梯价、优化 TM 自动回复话术与响应时效",
                )
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

    if sget("ad_stop"):
        prog.empty()
        status_ph.info("⏹ 已中止，已完成批次结果保留。")
        sset("ad_running", False)
        _finalize(partial=True)
        st.rerun()

    if sget("ad_finished"):
        prog.progress(1.0, text="全部批次完成")
        sset("ad_running", False)
        _finalize(partial=False)
        st.rerun()

    st.rerun()


def _finalize(partial: bool = False) -> None:
    df = sget("ad_df")
    results = sget("ad_batch_results", {}) or {}
    if partial and not results:
        return
    out = merge_results(df, results, id_col="产品ID", allowed=AD_TAGS)
    sset("ad_result", out)
    sset("ad_running", False)


# ==========================================================
# ④ 结果区
# ==========================================================

def _section_result() -> None:
    st.markdown("### ④ 诊断结果")

    # ⚠️ 必须在 tag_filter（multiselect）实例化之前执行
    apply_pending("ad_tag_pending", "ad_tag_filter", overwrite=True)

    out = sget("ad_result")
    if out is None:
        st.caption("尚未生成诊断结果。完成上方三步后自动显示。")
        return

    info = sget("ad_local_info", {})
    if info.get("degraded"):
        st.error("⚠️ 广告行业基准不可用，以下判定已降级为「店铺内部相对对比」，非行业标准。")
    ts = info.get("thr_source", {})
    for k in ("ctr", "lead_rate"):
        if ts.get(k) and not str(ts[k]).startswith("AI"):
            st.warning(f"⚠️ AI 未给出 {k} 阈值，已使用系统默认判定。")
    for note in info.get("notes", []):
        st.caption(f"ℹ️ {note}")
    st.caption(f"规则模式：{info.get('mode', '—')}")

    total = len(out)
    counts = count_tags(out, AD_TAGS)
    render_tag_summary_bar(counts, total, ai_covered_count(out), AD_TAGS)

    f1, f2, f3 = st.columns([2, 1.4, 1])
    with f1:
        tags = tag_filter("ad_tag_filter", AD_TAGS)
    with f2:
        sources = source_filter("ad_source_filter")
    with f3:
        if st.button("🔥 只看烧钱品", key="btn_only_burn", width="stretch"):
            # multiselect 本轮已实例化，只能暂存后下一轮落盘
            set_pending("ad_tag_pending", [AD_TAG_BURN])
            st.rerun()

    sort_key = st.selectbox("排序方式", options=["花费降序", "商机量降序", "曝光降序", "CPC 降序"],
                            index=0, key="ad_sort_key")
    _sort_map = {"花费降序": ("花费", False), "商机量降序": ("商机量", False),
                 "曝光降序": ("曝光量", False), "CPC 降序": ("CPC_计算", False)}
    col, asc = _sort_map.get(sort_key, ("花费", False))

    view = out.copy()
    if tags:
        view = view[view[COL_TAG].isin(tags)]
    if sources:
        view = view[view[COL_SOURCE].isin(sources)]
    if col in view.columns:
        view = view.sort_values(col, ascending=asc, na_position="last")

    if view.empty:
        st.info("当前筛选条件下没有数据。")
        return

    disp = view.copy()
    disp["商品信息"] = truncate_series(disp["商品信息"], MAX_DISPLAY_NAME_LEN)
    disp["花费"] = disp["花费"].map(fmt_money)
    disp["商机成本"] = disp["商机成本"].map(fmt_money)
    disp["点击率"] = disp["CTR_计算"].map(fmt_pct)
    disp["商机转化率"] = disp["商机转化率_计算"].map(fmt_pct)
    cols = [c for c in RESULT_COLS if c in disp.columns]
    disp = disp[cols]

    render_rows = render_rows_selector("ad_render_rows", len(disp))
    render_result_table(disp, RED_PLAINS, GOLD_PLAINS, render_rows=render_rows)

    std_text = (sget("ad_std") or IndustryStandard()).to_text()
    download_report_button(out[[c for c in RESULT_COLS if c in out.columns]],
                           std_text, "广告诊断报告", RED_PLAINS, GOLD_PLAINS,
                           label=f"⬇️ 下载广告诊断报告（Excel，全量 {len(out)} 行）")
