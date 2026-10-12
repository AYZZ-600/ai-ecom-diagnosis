"""
全局常量集中管理（对应 PRD §9 附录）

⚠️ 设计红线（PRD §3.4）：
本文件中【禁止】出现任何行业基准数值（如行业点击率、行业 CPC、行业均值倍数等）。
此文件只允许包含：
  ① 字段与口径定义
  ② 标签枚举
  ③ 展示/性能控制类常量
  ④ AI 未给出阈值时的保守兜底值（DEFAULT_LOW_RATIO，且必须在 UI 标注"系统默认"）
所有行业基准值必须来自 AI 联网调研返回的 metrics / thresholds。
"""

# ============ 千问 API ============
QWEN_MODEL = "qwen-plus"            # 可选：qwen-plus / qwen-max / qwen-turbo
QWEN_MODEL_CANDIDATES = ["qwen-plus", "qwen-max", "qwen-turbo"]
API_TIMEOUT_SECONDS = 120           # 联网搜索单次可能 20~40s，超时给足
RETRY_TIMES = 1                     # 单批失败重试次数
RETRY_SLEEP = 2                     # 重试间隔（秒）
RETRY_BACKOFF_BASE = 2              # 429 退避基数：2s -> 4s -> 8s

# ============ 文件 / Sheet ============
SHEET_PRODUCT = "产品分析"
DEFAULT_HEADER_ROW_PRODUCT = 6      # 表头在第 6 行 -> skiprows = 6 - 1 = 5
COLS_PRODUCT_REQUIRED = ["产品ID", "产品名称", "搜索曝光次数", "搜索点击次数", "询盘个数"]
COLS_PRODUCT_OPTIONAL = [
    "搜索点击率", "询盘率", "访问人数", "询盘人数",
    "TM咨询人数",            # ★ 商机口径必需：商机数 = 询盘个数 + TM咨询人数
    "提交订单个数", "收藏人数",
    "是否橱窗", "是否P4P", "产品负责人",
]

SHEET_AD = "sheet1"
DEFAULT_HEADER_ROW_AD = 1           # 表头在第 1 行 -> skiprows = 1 - 1 = 0
COLS_AD_REQUIRED = ["商品信息", "产品ID", "花费", "商机量", "商机成本", "点击量", "曝光量", "点击率"]
COLS_AD_OPTIONAL = ["点击成本", "商机转化率", "L1+商机量", "L1+点击量", "L1+买家点击占比", "询盘量", "订单量", "日期", "TM咨询量"]

# 可用物料 na_values
NA_TOKENS = ["-", "—", "–", "", "nan", "NaN", "None", "null"]

# ============ 规模控制（行数全部动态计算，禁止硬编码具体行数） ============
# 说明：N <= AI_FULL_LIMIT 走「全量模式」（所有行分批送 AI）；
#       N  > AI_FULL_LIMIT 走「混合模式」（本地规则全量打标 + AI 精修 Top-K）
# 已由 200 上调至 500（用户确认）：真实样例 343 行时应全部由 AI 精修，标签质量更高；
# 同时「AI 精修上限」保留为 UI 可调（50~2000），应对 2000+ 场景。
AI_FULL_LIMIT = 500
AI_TOP_K = 500                      # AI 精修上限默认值，UI 可调
AI_TOP_K_MIN = 50
AI_TOP_K_MAX = 2000
MAX_ROWS_PER_CALL = 50              # 单批最大行数（名称过长时自适应降为 30）
MAX_ROWS_PER_CALL_LONGNAME = 30
CONCURRENCY = 3                     # 并发批数（遇 429 自动降为 1）
CONCURRENCY_MIN = 1

# ============ 兜底阈值（仅当 AI 未返回 thresholds 时使用，UI 必须标注"系统默认"） ============
DEFAULT_LOW_RATIO = 0.7

# ============ 品类词词典 ============
# 用途：从标题里抽出商品的【核心品类词】，用于
#   ① 逐行喂给 AI，要求其建议沿用该词
#   ② 程序化兜底：AI 建议里若出现其它品类词，自动替换回正确的（防止改变产品定义）
# 规则：长词在前，优先匹配更具体的表述
CATEGORY_LEXICON = [
    "shower chair", "shower seat", "bench seat", "shower bench", "bath bench",
    "transfer bench", "commode chair", "bath lift", "toilet seat",
    "grab bar", "handrail", "safety rail", "support bar", "grab rail",
    "baseboard", "skirting board", "skirting", "wall panel", "shower panel",
    "wall sticker", "wallpaper", "wall paper",
    "shower head", "faucet", "towel rack", "corner shelf", "step stool",
]

# ============ 文本控制 ============
# 用户反馈：15 字限制导致建议说不清楚，已放开。
# 这里仅保留一个"防病态输出"的安全上限（AI 偶尔会输出整段话），正常建议 30~60 字。
MAX_SUGGESTION_LEN = 200
MAX_NAME_LEN = 60                   # 产品名称送 AI 前截断长度
MAX_DISPLAY_NAME_LEN = 40           # 页面展示时产品名称截断长度
CLASSIFY_SAMPLE = 20                # 类目自动识别的标题抽样条数

# ============ 渲染性能 ============
MAX_STYLE_ROWS = 1000               # Styler 逐行 apply 的性能红线
DEFAULT_RENDER_ROWS = 500           # 默认渲染行数
RENDER_ROWS_OPTIONS = [100, 500, 2000, 10 ** 6]   # 最后一项表示"全部"
EXCEL_ENGINE_SWITCH_ROWS = 1000     # >1000 行用 xlsxwriter，否则 openpyxl
TABLE_HEIGHT = 520

# ============ 高亮配色 ============
HIGHLIGHT_RED = "#FFE6E6"
HIGHLIGHT_GREEN = "#E8F5E9"
HIGHLIGHT_GOLD = "#FFF6E0"

# ============ 商品诊断标签枚举（5 个，最后一项必须是「正常」用于兜底） ============
TAG_HIGH_IMP_LOW_CTR = "🔴 高曝光低点击"
# ★ 用户反馈：核心判据由「询盘」改为「商机」（买家多数走 TM 而非询盘）
TAG_HIGH_CLK_LOW_OPP = "🟠 高点击低商机"
# ★ 用户反馈：高曝光+极低点击率要区分「主图不行」与「关键词写错导致流量不精准」
TAG_MIS_TRAFFIC = "🟣 流量不精准（关键词错配）"
TAG_GOOD = "🟢 表现良好"
TAG_NORMAL = "⚪ 正常"
PRODUCT_TAGS = [TAG_HIGH_IMP_LOW_CTR, TAG_HIGH_CLK_LOW_OPP, TAG_MIS_TRAFFIC,
                TAG_GOOD, TAG_NORMAL]
# 兼容旧名（避免其它引用处报错）
TAG_HIGH_CLK_LOW_INQ = TAG_HIGH_CLK_LOW_OPP

# ============ 广告诊断标签枚举（7 个，最后一项必须是「正常」用于兜底） ============
AD_TAG_BURN = "🔴 烧钱品"
AD_TAG_GOLD = "🟡 黄金品"
AD_TAG_TRAFFIC = "🔵 流量品"
AD_TAG_USELESS = "⚪ 鸡肋品"
# ★ 用户反馈：B2B 冷启动期花几块钱测词很正常，花费低+点击少不应直接判烧钱品
AD_TAG_TESTING = "🧪 测试中（样本不足）"
# ★ 用户反馈：花费为 0 是"根本没投放"，不是"正常"
AD_TAG_NOT_RUNNING = "⚫ 未投放（待激活）"
AD_TAG_NORMAL = "🟢 正常"
AD_TAGS = [AD_TAG_BURN, AD_TAG_GOLD, AD_TAG_TRAFFIC, AD_TAG_USELESS,
           AD_TAG_TESTING, AD_TAG_NOT_RUNNING, AD_TAG_NORMAL]

# 样本不足门槛：花费 < AD_MIN_COST_FOR_JUDGE 且 点击 < AD_MIN_CLICKS_FOR_JUDGE
# -> 判为「测试中（样本不足）」，不参与烧钱品/鸡肋品等定性判断
AD_MIN_COST_FOR_JUDGE = 50.0
AD_MIN_CLICKS_FOR_JUDGE = 10

# ============ 标签来源枚举 ============
SOURCE_AI = "AI"
SOURCE_LOCAL = "本地规则"
SOURCE_LOCAL_AI_FAIL = "本地规则（AI 返回异常）"
SOURCE_LOCAL_AI_ERROR = "本地规则（AI 服务异常）"

# ============ 结果表列名 ============
COL_TAG = "诊断标签"
COL_SOURCE = "标签来源"
COL_REASON = "诊断理由"
COL_SUGGESTION = "优化建议"
# 纯文本标签（去 emoji），用于 Excel 条件格式匹配（Excel 中 emoji 为代理对，直接 SEARCH 可能失败）
COL_TAG_PLAIN = "标签文本"

TAG_PLAIN_MAP = {
    "🔴": "", "🟠": "", "🟢": "", "🟡": "", "🔵": "", "⚪": "",
}
