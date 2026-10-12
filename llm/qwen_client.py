"""千问（DashScope）API 封装：联网搜索 / 普通推理 / JSON 输出。

对应 PRD §6.3。要点：
- enable_search 仅在 Step 1 行业调研开启；Step 3 诊断阶段关闭（省 token，避免搜索干扰判定）。
- enable_search 与 response_format=json_object 不可同时使用，故诊断阶段用「提示词强约束 + 正则兜底」。
- 429 / Throttling 自动退避重试。
- 搜索引用使用 getattr 安全取值，不硬编码字段名。
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from dashscope import Generation
import dashscope

try:  # 本地脚本（非 Streamlit）运行时没有 st.secrets
    import streamlit as st
except Exception:  # pragma: no cover
    st = None

def _read_key() -> str:
    """只从 st.secrets / 环境变量读取，禁止硬编码。"""
    key = ""
    if st is not None:
        try:
            key = st.secrets.get("DASHSCOPE_API_KEY", "") or ""
        except Exception:
            key = ""
    if not key:
        key = os.getenv("DASHSCOPE_API_KEY", "") or ""
    return key.strip()


dashscope.api_key = _read_key()

from config import (  # noqa: E402
    QWEN_MODEL,
    API_TIMEOUT_SECONDS,
    RETRY_TIMES,
    RETRY_SLEEP,
    RETRY_BACKOFF_BASE,
)


class LLMError(Exception):
    """API 调用失败。携带 help 信息供 UI 友好展示。"""


class MissingAPIKeyError(LLMError):
    pass


_URL_RE = re.compile(r"https?://[^\s\"'）)，。；、\]】]+")


@dataclass
class LLMResponse:
    text: str
    model: str = ""
    search_info: list = field(default_factory=list)  # 结构化引用（当前接口未返回，通常为空）
    request_id: str = ""
    search_count: int = 0            # usage.plugins.search.count —— 本次实际联网检索次数
    input_tokens: int = 0
    output_tokens: int = 0
    searched: bool = False           # 是否真正触发了联网搜索插件


def _extract_usage(resp) -> dict:
    """从 usage 中提取 tokens 与联网检索次数。

    实测：DashScope 联网返回的 usage 形如
        {"input_tokens":4136,"output_tokens":800,"total_tokens":4936,
         "plugins":{"search":{"count":1,"strategy":"turbo"}}}
    「plugins.search」存在即代表本次调用了联网检索插件，可用于 UI 透明度展示。
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {}
    if not isinstance(usage, dict):
        usage = dict(usage) if hasattr(usage, "items") else {}
    plugins = usage.get("plugins") or {}
    if not isinstance(plugins, dict):
        plugins = {}
    search = plugins.get("search") or {}
    if not isinstance(search, dict):
        search = {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "search_count": int(search.get("count") or 0),
        "search_strategy": str(search.get("strategy") or ""),
    }


def extract_text_sources(text: str) -> list[dict]:
    """兜底：从正文中抽取 AI 自己写出的来源网址。

    原因：当前 DashScope 接口未返回结构化引用列表（search_info 为空），
    因此「来源」只能依赖提示词要求 AI 在 JSON 的 source 字段或正文中写出。
    """
    seen, out = set(), []
    for u in _URL_RE.findall(text or ""):
        u = u.rstrip(".,;:)]}>")
        if u and u not in seen:
            seen.add(u)
            out.append({"title": "", "url": u})
    return out


def refresh_api_key() -> None:
    """密钥变更后刷新 SDK 全局配置。"""
    dashscope.api_key = _read_key()


def get_api_key() -> str:
    return _read_key()


def has_api_key() -> bool:
    return bool(_read_key())


def _extract_search_info(resp) -> list:
    """安全提取联网搜索引用。不同版本 SDK 字段名不同，不做硬编码假设。

    返回形如 [{"title": ..., "url": ...}, ...]，取不到则返回空列表。
    """
    info = getattr(resp, "search_info", None)
    out: list[dict] = []
    if not info:
        return out
    # DashScope 新版：SearchInfo(search_results=[SearchResult(title, url, ...), ...])
    results = getattr(info, "search_results", None) if not isinstance(info, dict) else info.get("search_results")
    if results is None:
        return out
    for r in results:
        if isinstance(r, dict):
            title, url = r.get("title", ""), r.get("url", "")
        else:
            title = getattr(r, "title", "") or ""
            url = getattr(r, "url", "") or ""
        if title or url:
            out.append({"title": title, "url": url})
    return out


def qwen_chat(
    messages: list[dict],
    model: str = QWEN_MODEL,
    enable_search: bool = False,
    temperature: float = 0.2,
    max_tokens: int = 2000,
    retries: int = RETRY_TIMES,
) -> LLMResponse:
    """调用千问，返回 LLMResponse。失败抛 LLMError。

    enable_search=True  -> 开启阿里联网搜索（服务端先检索公网再作答）
    enable_search=False -> 纯模型推理（诊断阶段使用，更快更便宜）
    """
    if not has_api_key():
        raise MissingAPIKeyError(
            "未配置 DASHSCOPE_API_KEY。请在 .streamlit/secrets.toml 中填写，"
            "或设置环境变量 DASHSCOPE_API_KEY。"
        )

    dashscope.api_key = _read_key()
    dashscope.request_timeout = API_TIMEOUT_SECONDS

    last_err = ""
    for attempt in range(retries + 1):
        try:
            resp = Generation.call(
                model=model,
                messages=messages,
                enable_search=enable_search,  # ★ 联网搜索开关
                result_format="message",
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if resp.status_code == 200:
                choice = resp.output.choices[0]
                content = getattr(choice.message, "content", None) or ""
                usage = _extract_usage(resp)
                search_info = _extract_search_info(resp)
                return LLMResponse(
                    text=content.strip(),
                    model=model,
                    search_info=search_info,
                    request_id=getattr(resp, "request_id", "") or "",
                    search_count=usage.get("search_count", 0),
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    searched=usage.get("search_count", 0) > 0,
                )
            last_err = f"[{resp.status_code}] {getattr(resp, 'code', '')} - {getattr(resp, 'message', '')}"
        except Exception as e:  # 网络异常 / SDK 异常
            last_err = str(e)

        # 429 / Throttling 指数退避，其余固定间隔
        if "429" in last_err or "Throttling" in last_err or "throttl" in last_err.lower():
            time.sleep(RETRY_BACKOFF_BASE ** (attempt + 1))
        else:
            time.sleep(RETRY_SLEEP)

    raise LLMError(f"千问 API 调用失败（已重试 {retries} 次）：{last_err}")


def qwen_text(*args, **kwargs) -> str:
    """便捷方法：只取正文文本。"""
    return qwen_chat(*args, **kwargs).text
