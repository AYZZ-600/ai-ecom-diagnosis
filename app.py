"""AI 电商经营诊断平台 —— Streamlit 入口（PRD §5）。

本地运行：streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from llm import has_api_key
from modules import product_tab, ad_tab

st.set_page_config(
    page_title="AI电商经营诊断平台",
    page_icon="⛺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- 全局样式 ----------
st.markdown("""
<style>
  .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
  h1 { font-weight: 700; letter-spacing: -0.5px; }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] { height: 44px; padding: 0 22px; border-radius: 8px 8px 0 0; }
  div[data-testid="stMetric"] {
      border: 1px solid #EDEFF2; border-radius: 10px; padding: 10px 12px;
      background: #FFFFFF;
  }
  .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("⛺ AI 电商经营诊断平台")
st.caption("阿里国际站 B2B 建材类目 · 行业基准 × 店铺数据 智能诊断　｜　"
           "标准由 AI 联网自主调研，人不参与定标准")

if not has_api_key():
    st.error(
        "❌ 未检测到 API Key，AI 功能不可用。\n\n"
        "请在项目根目录创建 `.streamlit/secrets.toml`：\n"
        "```toml\nDASHSCOPE_API_KEY = \"sk-...\"\n```\n"
        "（云端部署请在 Streamlit Cloud → Settings → Secrets 中配置）"
    )

tab1, tab2 = st.tabs(["📦 商品诊断", "📢 广告诊断"])

with tab1:
    try:
        product_tab.render_tab()
    except Exception as e:
        st.exception(e)

with tab2:
    try:
        ad_tab.render_tab()
    except Exception as e:
        st.exception(e)

st.divider()
st.caption(
    "⚠️ 上传数据将用于 AI 分析，请勿上传含客户隐私信息的表单　|　"
    "行业基准由千问联网检索与自主判断生成，仅供参考，请结合店铺实际经营阶段判断。"
)
