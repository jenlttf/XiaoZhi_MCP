import sys
import os
import json
import asyncio
import logging
import subprocess
import signal
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from pathlib import Path
from datetime import datetime

# ==========================================
# 0. 依賴檢查
# ==========================================
try:
    from mcp.server.fastmcp import FastMCP
    import websockets
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    from google.api_core import exceptions
    from openai import OpenAI
    try:
        from openai import AsyncOpenAI
    except Exception:
        AsyncOpenAI = None

    try:
        import opencc
    except ImportError:
        opencc = None
except ImportError as e:
    sys.stderr.write(f"❌ 缺少必要依賴！請執行: pip install google-genai websockets fastmcp python-dotenv opencc-python-reimplemented openai\n")
    sys.exit(1)

# ==========================================
# 1. 配置與核心類別
# ==========================================
CONFIG_PATH = Path.home() / ".xiaozhi_mcp_config.json"
KEY_FILE_PATH = os.path.join(os.sep, "home", "ccc", ".config", "gemini", "api_key")
OPENAI_KEY_FILE_PATH = os.path.join(os.sep, "home", "ccc", ".config", "openai", "api_key")

# OpenAI 省錢預設
USD_TO_TWD = 30.0
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_MAX_TOOL_CALLS = 1
OPENAI_MAX_OUTPUT_TOKENS = 700
OPENAI_TEMPERATURE = 0.2

# OpenAI 計價估算 (以 Standard 價格)
# gpt-4o-mini: input 0.15, cached input 0.075, output 0.60 (USD per 1M tokens)
# web search: 10 USD per 1k calls
# gpt-4o-mini 使用非預覽 web search 時，search content 以每次 8000 input tokens 固定計費
OPENAI_GPT4O_MINI_USD_PER_1M = {"input": 0.15, "cached_input": 0.075, "output": 0.60}
OPENAI_WEB_SEARCH_USD_PER_CALL = 10.0 / 1000.0
OPENAI_WEB_SEARCH_FIXED_SEARCH_TOKENS = 8000

def build_search_prompt(query_text: str) -> str:
    """統一的搜尋提示詞，供 Gemini 與 OpenAI 共用"""
    return (
        "未使用網頁搜尋工具，直接由AI自行回答時，回答濃縮成300字以內，但不能少於250字。回答要完整。"
        "非新聞類的搜尋，回答濃縮成300字以內，但不能少於250字。回答要完整。不用註記網頁來源。"
        "請先使用網頁搜尋工具，回答下列查詢。若屬新聞類，優先挑選最近 48 小時內最重要的 5 則，"
        "每則只包含1 到 2 句重點。不用註記網頁來源。\n\n"
        f"查詢：{query_text}"
    )


def build_hybrid_prompt(query_text: str) -> str:
    """混合模式：能直答就直答，不確定再用網頁搜尋工具。"""
    return (
        "請先嘗試不使用網頁搜尋工具，直接回答下列查詢。"
        "若你不確定、需要即時資訊、或問題屬新聞時，再使用網頁搜尋工具補強。"
        "未使用網頁搜尋工具時：回答濃縮成300字以內，但不能少於250字，且內容要完整。"
        "使用網頁搜尋工具且非新聞時：回答同樣濃縮成300字以內，但不能少於250字，內容要完整，不用註記網頁來源。"
        "若屬新聞類且使用網頁搜尋：優先挑選最近48小時內最重要的5則，每則只包含1到2句重點，不用註記網頁來源。\n\n"
        f"查詢：{query_text}"
    )


def load_config():
    config = {
        "MCP_ENDPOINT": "wss://api.xiaozhi.me/mcp/?token=您的Token",
        "GOOGLE_API_KEY": "",
        "OPENAI_API_KEY": "",
    }
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            saved = json.load(f)
            config.update(saved)
    except: pass
    
    if os.path.exists(KEY_FILE_PATH) and not config["GOOGLE_API_KEY"]:
        try:
            with open(KEY_FILE_PATH, "r") as f:
                config["GOOGLE_API_KEY"] = f.read().strip()
        except: pass

    if os.path.exists(OPENAI_KEY_FILE_PATH) and not config["OPENAI_API_KEY"]:
        try:
            with open(OPENAI_KEY_FILE_PATH, "r") as f:
                config["OPENAI_API_KEY"] = f.read().strip()
        except:
            pass
    return config

def save_config(config):
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

class GeminiBlindManager:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self._lock = asyncio.Lock()
        self.priority_models = []
        self.current_model_name = None

    def log(self, msg):
        sys.stderr.write(f"[GUI_LOG]{msg}\n")
        sys.stderr.flush()

    async def _test_model_basic(self, model_name):
        """第一關：基礎連線與工具綁定測試"""
        try:
            test_tool = types.Tool(google_search=types.GoogleSearch())
            config = types.GenerateContentConfig(
                tools=[test_tool],
                max_output_tokens=5
            )
            await self.client.aio.models.generate_content(
                model=model_name, 
                contents="Hi", 
                config=config
            )
            return True, None
        except Exception as e:
            err_msg = str(e)
            
            # [404] 針對找不到模型的情況，提供明確訊息
            if "404" in err_msg or "NOT_FOUND" in err_msg:
                return False, "模型不存在 (需使用具體版號)"

            if "429" in err_msg: return False, "配額已滿 (429)"
            if "503" in err_msg: return False, "服務忙碌 (503)"
            if "not support" in err_msg or "Invalid" in err_msg or "400" in err_msg:
                return False, "不支援搜尋工具"
            return False, f"連線錯誤: {err_msg[:40]}..."

    async def _test_model_real_search(self, model_name):
        """第二關：實彈搜尋測試"""
        try:
            test_tool = types.Tool(google_search=types.GoogleSearch())
            config = types.GenerateContentConfig(
                tools=[test_tool],
                max_output_tokens=50 
            )
            response = await self.client.aio.models.generate_content(
                model=model_name, 
                contents="Search news about 2026", 
                config=config
            )
            
            if response.candidates and response.candidates[0].grounding_metadata:
                return True, None
            
            return True, "無搜尋數據但回應正常"
            
        except Exception as e:
            return False, f"搜尋功能異常: {str(e)[:40]}..."

    async def initialize_models(self):
        self.log("🔍 系統啟動：正在掃描 (強制使用 001/002 精確版號)...")
        try:
            all_models = list(self.client.models.list())
            candidates = []
            
            black_keywords = [
                "image", "vision", "embedding", "creation", 
                "bison", "gecko", "text-", "answer-", "tutorial", 
                "lite", "8b", "nano",
                "robotics", "er-", "tts", "audio", "speech", "realtime", 
                "thinking", "learn", "aqa"
            ]

            for m in all_models:
                name = m.name.replace("models/", "")
                if "gemini" not in name: continue
                if any(bk in name for bk in black_keywords): continue
                candidates.append(name)

            # =========================================================
            # [核心修正] 強制注入「帶版號」的 1.5 模型
            # 解決通用別名 (gemini-1.5-flash) 報 404 的問題
            # =========================================================
            specific_versions = [
                "gemini-1.5-flash-002", # 最新穩定版
                "gemini-1.5-flash-001", # 舊穩定版 (通常最保險)
                "gemini-1.5-pro-002",
                "gemini-1.5-pro-001"
            ]
            
            for ver in specific_versions:
                if ver not in candidates:
                    candidates.append(ver)
                    # self.log(f"   (注入精確版號: {ver})")

            # 排序優化
            def score(n):
                s = 0
                if "2.0-flash" in n and "exp" not in n: s += 5000 
                if "2.0-flash" in n and "exp" in n: s += 4000     
                
                # 1.5 Flash 002 > 001
                if "1.5-flash-002" in n: s += 3500
                if "1.5-flash-001" in n: s += 3400
                if "1.5-flash" in n: s += 3000 # 通用別名(如果有)

                if "1.5-pro" in n: s += 2000
                if "latest" in n: s += 100
                return s

            sorted_candidates = sorted(candidates, key=score, reverse=True)
            self.log(f"📋 測試名單: {sorted_candidates[:6]}...")

            valid_model = None
            failed_models = set()

            for model in sorted_candidates:
                self.log(f"   ...檢測: {model}")
                
                # [Phase 1] 基礎連線
                is_basic_ok, basic_msg = await self._test_model_basic(model)
                if not is_basic_ok:
                    self.log(f"      ⚠️ 第一關失敗 ({basic_msg}) -> 跳過")
                    failed_models.add(model)
                    continue

                # [Phase 2] 實彈搜尋
                self.log(f"      🔹 連線成功，嘗試搜尋新聞...")
                is_search_ok, search_msg = await self._test_model_real_search(model)
                
                if is_search_ok:
                    self.log(f"      ✅ 驗證通過! ({model})")
                    valid_model = model
                    break 
                else:
                    self.log(f"      ⚠️ 第二關失敗 ({search_msg})")
                    failed_models.add(model)

            # 設定結果
            if valid_model:
                backups = [m for m in sorted_candidates 
                           if m != valid_model and m not in failed_models]
                self.priority_models = [valid_model] + backups
                self.current_model_name = valid_model
                self.log(f"ℹ️ 最終選定: {valid_model}")
            else:
                self.log("❌ 嚴重警告：所有模型 (含 1.5-001/002) 皆不可用")
                self.priority_models = []
                self.current_model_name = None

        except Exception as e:
            self.log(f"❌ 初始化失敗: {e}")
            # 保底使用最老牌的 001
            self.priority_models = ["gemini-1.5-flash-001"]
            self.current_model_name = "gemini-1.5-flash-001"

    async def generate_content_async(self, prompt, tools=None):
        if not self.current_model_name or not self.priority_models:
             raise Exception("無可用模型")

        async with self._lock: 
            start_model = self.current_model_name
        
        try_list = []
        if start_model and start_model in self.priority_models:
            try_list.append(start_model)
        for m in self.priority_models:
            if m not in try_list: try_list.append(m)
        try_list = try_list[:3]

        for model in try_list:
            try:
                config = types.GenerateContentConfig(tools=tools, temperature=0.3)
                response = await self.client.aio.models.generate_content(
                    model=model, contents=prompt, config=config
                )
                
                if self.current_model_name != model:
                    self.current_model_name = model
                    self.log(f"ℹ️ 切換至: {model}")
                return response

            except Exception as e:
                err_msg = str(e)
                short_err = "API 錯誤"
                if "429" in err_msg: short_err = "配額不足"
                elif "503" in err_msg: short_err = "服務忙碌"
                elif "404" in err_msg: short_err = "模型找不到"
                
                self.log(f"⚠️ {model} 執行失敗 ({short_err})，嘗試下一個...")
                continue
        
        raise Exception("所有嘗試皆失敗")

class OpenAIWebSearchManager:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.client = None
        self._init_client()

    def _init_client(self):
        if not self.api_key:
            cfg = load_config()
            self.api_key = cfg.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""

        if not self.api_key:
            raise Exception("未配置 OpenAI API Key")

        # 優先用 AsyncOpenAI，沒有就退回同步 client
        if AsyncOpenAI is not None:
            self.client = AsyncOpenAI(api_key=self.api_key)
        else:
            self.client = OpenAI(api_key=self.api_key)

    def log(self, msg):
        sys.stderr.write(f"[GUI_LOG]{msg}\n")
        sys.stderr.flush()

    @staticmethod
    def _to_dict(obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))

    @staticmethod
    def _count_web_search_calls(resp_dict: dict) -> int:
        out = resp_dict.get("output", []) or []
        n = 0
        for item in out:
            if isinstance(item, dict) and item.get("type") == "web_search_call":
                n += 1
        return n

    @staticmethod
    def _extract_usage(resp_dict: dict) -> tuple[int, int, int]:
        usage = resp_dict.get("usage", {}) or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)

        cached_tokens = 0
        itd = usage.get("input_tokens_details")
        if isinstance(itd, dict):
            cached_tokens = int(itd.get("cached_tokens", 0) or 0)

        return input_tokens, output_tokens, cached_tokens

    @staticmethod
    def _estimate_cost_usd(input_tokens: int, output_tokens: int, cached_tokens: int, web_calls: int) -> tuple[float, float]:
        # token cost
        p_in = OPENAI_GPT4O_MINI_USD_PER_1M["input"] / 1_000_000.0
        p_cached = OPENAI_GPT4O_MINI_USD_PER_1M["cached_input"] / 1_000_000.0
        p_out = OPENAI_GPT4O_MINI_USD_PER_1M["output"] / 1_000_000.0

        non_cached_in = max(0, input_tokens - cached_tokens)
        token_cost = (non_cached_in * p_in) + (cached_tokens * p_cached) + (output_tokens * p_out)

        # tool cost
        tool_cost = web_calls * OPENAI_WEB_SEARCH_USD_PER_CALL

        total = token_cost + tool_cost
        return total, total * USD_TO_TWD


    @staticmethod
    def _needs_web(q: str) -> bool:
        q = (q or "").strip().lower()
        if not q:
            return False
        keywords = [
            "新聞", "快訊", "最新", "今天", "昨日", "昨天", "這週", "本週", "剛剛",
            "股價", "匯率", "價格", "報價", "天氣", "氣溫", "降雨", "台風",
            "賽程", "比分", "名單", "得獎", "公告", "招標", "標案", "釋疑",
            "修正", "更新", "版本", "費用", "方案價格", "上市", "下市", "停牌",
            "營業時間", "門市", "附近"
        ]
        return any(k.lower() in q for k in keywords)

    @staticmethod
    def _looks_uncertain(ans: str) -> bool:
        if not ans:
            return True
        flags = [
            "我不確定", "不確定", "可能已更新", "可能有變動", "需要查證",
            "建議查詢", "以官方公告為準", "我無法確認", "需以最新資料",
            "【需要搜尋】"
        ]
        return any(f in ans for f in flags)

    @staticmethod
    def _strip_prefix(q: str) -> tuple[str, bool, bool]:
        s = (q or "").strip()
        force_web = False
        force_no_web = False

        if s.startswith("強制搜尋"):
            force_web = True
            s = s[len("強制搜尋"):].strip()
        if s.startswith("強制不搜"):
            force_no_web = True
            s = s[len("強制不搜"):].strip()

        return s, force_web, force_no_web

    async def direct_answer(self, query_text: str) -> str:
        prompt = (
            "請直接回答下列問題。"
            "回答濃縮成300字以內，但不能少於250字。回答要完整。"
            "若你判斷必須依賴最新資訊或外部查證才可靠，請在開頭輸出【需要搜尋】並用一句話說明原因。"
            "其餘情況不要提搜尋。\n\n"
            f"問題：{query_text}"
        )

        kwargs = dict(
            model=OPENAI_MODEL,
            input=prompt,
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
            temperature=OPENAI_TEMPERATURE,
        )

        if AsyncOpenAI is not None and hasattr(self.client, "responses"):
            resp = await self.client.responses.create(**kwargs)
        else:
            resp = await asyncio.to_thread(self.client.responses.create, **kwargs)

        text = getattr(resp, "output_text", None) or ""
        resp_dict = self._to_dict(resp)
        input_tokens, output_tokens, cached_tokens = self._extract_usage(resp_dict)

        total_usd, total_twd = self._estimate_cost_usd(input_tokens, output_tokens, cached_tokens, 0)
        self.log(f"🧠 OpenAI 直答: {query_text} (模型: {OPENAI_MODEL})")
        self.log(f"💰 OpenAI 估算費用: USD {total_usd:.6f} , TWD {total_twd:.3f}  (input {input_tokens}, output {output_tokens}, web_calls 0)")

        return text.strip() or "直答完成但無內容"

    async def hybrid_answer(self, query_text: str) -> str:
        # 單次呼叫：先直答，必要時才讓模型自行使用 web_search tool，避免「直答不確定」再跑第二次造成逾時
        tz = "Asia" + chr(47) + "Taipei"

        tools = [{
            "type": "web_search",
            "user_location": {
                "type": "approximate",
                "country": "TW",
                "city": "Taipei",
                "region": "Taipei",
                "timezone": tz,
            }
        }]

        prompt = build_hybrid_prompt(query_text)

        kwargs = dict(
            model=OPENAI_MODEL,
            input=prompt,
            tools=tools,
            tool_choice="auto",
            max_tool_calls=OPENAI_MAX_TOOL_CALLS,
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
            temperature=OPENAI_TEMPERATURE,
        )

        if AsyncOpenAI is not None and hasattr(self.client, "responses"):
            resp = await self.client.responses.create(**kwargs)
        else:
            resp = await asyncio.to_thread(self.client.responses.create, **kwargs)

        text = getattr(resp, "output_text", None) or ""
        resp_dict = self._to_dict(resp)

        web_calls = self._count_web_search_calls(resp_dict)
        input_tokens, output_tokens, cached_tokens = self._extract_usage(resp_dict)

        # 若真的用了 web_search，做一次保守補強，避免 usage 低估導致費用估算偏小
        if web_calls > 0:
            fixed = OPENAI_WEB_SEARCH_FIXED_SEARCH_TOKENS * web_calls
            if input_tokens < int(0.5 * fixed):
                input_tokens = input_tokens + fixed

        total_usd, total_twd = self._estimate_cost_usd(input_tokens, output_tokens, cached_tokens, web_calls)
        self.log(f"🧠 OpenAI 混合: {query_text} (模型: {OPENAI_MODEL})")
        self.log(f"💰 OpenAI 估算費用: USD {total_usd:.6f} , TWD {total_twd:.3f}  (input {input_tokens}, output {output_tokens}, web_calls {web_calls})")

        return text.strip() or "完成但無內容"



    async def smart_answer(self, query_text: str) -> str:
        q, force_web, force_no_web = self._strip_prefix(query_text)

        if force_web:
            return await self.web_search(q)

        if force_no_web:
            return await self.direct_answer(q)

        if not self._needs_web(q):
            ans = await self.hybrid_answer(q)
            if (not self._looks_uncertain(ans)) and (not ans.startswith("【需要搜尋】")):
                return ans
            self.log("⚠️ OpenAI 混合模式仍不確定，改用強制搜尋補強")
            return await self.web_search(q)

        return await self.web_search(q)

    async def web_search(self, query_text: str) -> str:
        tz = "Asia" + chr(47) + "Taipei"

        tools = [{
            "type": "web_search",
            "user_location": {
                "type": "approximate",
                "country": "TW",
                "city": "Taipei",
                "region": "Taipei",
                "timezone": tz,
            }
        }]

        prompt = build_search_prompt(query_text)

        kwargs = dict(
            model=OPENAI_MODEL,
            input=prompt,
            tools=tools,
            tool_choice="auto",
            max_tool_calls=OPENAI_MAX_TOOL_CALLS,
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
            temperature=OPENAI_TEMPERATURE,
            include=["web_search_call.action.sources"],
        )

        # 發送請求
        if AsyncOpenAI is not None and hasattr(self.client, "responses"):
            resp = await self.client.responses.create(**kwargs)
        else:
            # 同步 client 退回 thread
            resp = await asyncio.to_thread(self.client.responses.create, **kwargs)

        text = getattr(resp, "output_text", None) or ""
        resp_dict = self._to_dict(resp)

        web_calls = self._count_web_search_calls(resp_dict)
        input_tokens, output_tokens, cached_tokens = self._extract_usage(resp_dict)

        # gpt-4o-mini 使用非預覽 web search 時，search content 以每次 8000 input tokens 固定計費
        # 有些情況 usage 可能未完整反映，這裡做保守補強
        if web_calls > 0:
            fixed = OPENAI_WEB_SEARCH_FIXED_SEARCH_TOKENS * web_calls
            if input_tokens < int(0.5 * fixed):
                input_tokens = input_tokens + fixed

        total_usd, total_twd = self._estimate_cost_usd(input_tokens, output_tokens, cached_tokens, web_calls)

        self.log(f"🔎 OpenAI 搜尋: {query_text} (模型: {OPENAI_MODEL})")
        self.log(f"💰 OpenAI 估算費用: USD {total_usd:.6f} , TWD {total_twd:.3f}  (input {input_tokens}, output {output_tokens}, web_calls {web_calls})")

        # preview
        if text:
            lines = [l for l in text.split("\n") if l.strip()]
            for line in lines:
                self.log(f"  {line}")

        return text or "搜尋完成但無內容"


# ==========================================
# 2. MCP 伺服器
# ==========================================
gemini_agent = None
openai_agent = None
cc_converter = None

def run_mcp_server():
    logging.getLogger("httpcore").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    logging.getLogger("google").setLevel(logging.CRITICAL)
    
    os.environ["PYTHONUNBUFFERED"] = "1"
    if sys.platform == 'win32':
        sys.stderr.reconfigure(encoding='utf-8')
        sys.stdout.reconfigure(encoding='utf-8')

    config = load_config()
    api_key = config.get("GOOGLE_API_KEY")
    
    # --- 1. 初始化 Gemini ---
    if api_key:
        temp_agent = GeminiBlindManager(api_key)
        try:
            asyncio.run(temp_agent.initialize_models())
            global gemini_agent
            gemini_agent = temp_agent
        except Exception as e:
            sys.stderr.write(f"[GUI_LOG]❌ 啟動檢查失敗: {e}\n")
    else:
        sys.stderr.write("[GUI_LOG]⚠️ 未檢測到 API Key\n")

    # --- 2. 初始化 OpenAI (備援預先載入) ---
    global openai_agent
    try:
        # 嘗試先建立實例，檢查 Key 是否存在與 Client 建立是否成功
        openai_agent = OpenAIWebSearchManager()
        sys.stderr.write("[GUI_LOG]✅ OpenAI 已就緒：備援已初始化\n")
        sys.stderr.flush()
    except Exception as e:
        # 若失敗(例如無 Key)，僅提示警告，不阻斷主程式，等到真正調用時再報錯
        sys.stderr.write(f"[GUI_LOG]⚠️ OpenAI 備援未啟用：{str(e)[:60]}\n")
        sys.stderr.flush()

    # --- 3. 啟動 MCP ---
    mcp = FastMCP("Xiaozhi-Gemini-Search")

    @mcp.tool(name="web_search")
    async def web_search(query_text: str) -> str:
        global gemini_agent, openai_agent, cc_converter

        final_query = query_text
        if opencc:
            try:
                if cc_converter is None:
                    cc_converter = opencc.OpenCC('s2twp')
                final_query = cc_converter.convert(query_text)
            except:
                pass

        # ----------------------------------------------------------
        # 先嘗試 Gemini (免費模型)
        # ----------------------------------------------------------
        try:
            if gemini_agent is None:
                config = load_config()
                api_key = config.get("GOOGLE_API_KEY")
                if api_key:
                    gemini_agent = GeminiBlindManager(api_key)
                    await gemini_agent.initialize_models()

            if gemini_agent and gemini_agent.current_model_name:
                sys.stderr.write(f"[GUI_LOG]🔎 搜尋: {final_query} (Gemini 模型: {gemini_agent.current_model_name})\n")
                sys.stderr.flush()

                tool = types.Tool(google_search=types.GoogleSearch())
                prompt = build_search_prompt(final_query)
                response = await gemini_agent.generate_content_async(prompt, tools=[tool])

                result_text = ""
                if response.text:
                    result_text = response.text
                elif response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.text:
                            result_text += part.text

                if not result_text:
                    result_text = "搜尋完成但無內容"

                sys.stderr.write(f"[GUI_LOG]✅ 完成 (Gemini 字數: {len(result_text)})\n")
                sys.stderr.flush()
                return result_text

        except Exception as e:
            # Gemini 任何失敗都改走 OpenAI
            sys.stderr.write(f"[GUI_LOG]⚠️ Gemini 不可用，改用 OpenAI：{str(e)[:80]}\n")
            sys.stderr.flush()

        # ----------------------------------------------------------
        # OpenAI fallback：省錢版 web_search (會顯示費用)
        # ----------------------------------------------------------
        try:
            # 雖然啟動時嘗試初始化過，但如果啟動時失敗(例如沒 Key)，這邊會再嘗試一次或直接報錯
            if openai_agent is None:
                openai_agent = OpenAIWebSearchManager()

            return await openai_agent.smart_answer(final_query)

        except Exception as e:
            err = f"搜尋錯誤: {e}"
            sys.stderr.write(f"[GUI_LOG]❌ {err}\n")
            sys.stderr.flush()
            return err

    mcp.run(transport="stdio")

# ==========================================
# 3. Pipe (連接器)
# ==========================================
def run_mcp_pipe():
    logging.basicConfig(level=logging.CRITICAL) 
    config = load_config()
    endpoint = os.environ.get("MCP_ENDPOINT", config.get("MCP_ENDPOINT"))
    if not endpoint or "token=" not in endpoint: return

    os.environ["PYTHONUNBUFFERED"] = "1"

    async def connect():
        uri = endpoint
        process = None
        try:
            process = subprocess.Popen(
                [sys.executable, "-u", os.path.abspath(__file__), "--server"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                encoding='utf-8', errors='replace', bufsize=0
            )
            
            async with websockets.connect(uri) as ws:
                async def pipe_ws():
                    while True:
                        msg = await ws.recv()
                        process.stdin.write(msg + '\n')
                        process.stdin.flush()

                async def pipe_out():
                    while True:
                        data = await asyncio.get_event_loop().run_in_executor(None, process.stdout.readline)
                        if not data: break
                        await ws.send(data)

                async def pipe_err():
                    while True:
                        data = await asyncio.get_event_loop().run_in_executor(None, process.stderr.readline)
                        if not data: break
                        sys.stderr.write(data)
                        sys.stderr.flush()

                await asyncio.gather(pipe_ws(), pipe_out(), pipe_err())
        except Exception as e:
            sys.stderr.write(f"[GUI_LOG]連接錯誤: {e}\n")
        finally:
            if process: process.terminate()

    async def retry_loop():
        while True:
            try: await connect()
            except: await asyncio.sleep(5)

    try: asyncio.run(retry_loop())
    except: pass

# ==========================================
# 4. GUI (介面)
# ==========================================
class MCPApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("小智 MCP (Gemini 優先，OpenAI 備援)")
        self.root.geometry("850x650")
        self.config = load_config()
        self.process = None
        self.msg_queue = queue.Queue()
        
        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.process_queue)

    def setup_ui(self):
        ttk.Label(self.root, text="🚀 小智聯網工具 (Gemini 優先，失敗自動改 OpenAI)", font=("微軟正黑體", 12, "bold")).pack(pady=10)
        
        frame = ttk.LabelFrame(self.root, text="設定")
        frame.pack(padx=10, fill="x")
        
        ttk.Label(frame, text="MCP:").grid(row=0, column=0)
        self.mcp_ent = ttk.Entry(frame, width=60)
        self.mcp_ent.insert(0, self.config.get("MCP_ENDPOINT", ""))
        self.mcp_ent.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Key:").grid(row=1, column=0)
        self.key_ent = ttk.Entry(frame, width=60)
        k = self.config.get("GOOGLE_API_KEY", "")
        if not k and os.path.exists(KEY_FILE_PATH): k = "(系統自動讀取)"
        self.key_ent.insert(0, k)
        self.key_ent.grid(row=1, column=1, padx=5, pady=5)

        btn_f = ttk.Frame(self.root)
        btn_f.pack(pady=10)
        ttk.Button(btn_f, text="保存設定", command=self.save).pack(side="left", padx=5)
        self.btn_run = ttk.Button(btn_f, text="啟動服務", command=self.toggle)
        self.btn_run.pack(side="left", padx=5)

        log_f = ttk.LabelFrame(self.root, text="系統監控日誌")
        log_f.pack(padx=10, fill="both", expand=True)
        self.log_txt = scrolledtext.ScrolledText(log_f, height=15, state='disabled', font=("Consolas", 9))
        self.log_txt.pack(padx=5, pady=5, fill="both", expand=True)
        
        ttk.Label(log_f, text="說明: 先嘗試 Gemini 具體版號模型，若全數不可用則自動切換 OpenAI web search，並在日誌顯示估算費用。", foreground="green").pack(anchor="w", padx=5)
        ttk.Label(log_f, text="OpenAI Key: 系統會從預設位置自動讀取，不提供手動設定。", foreground="gray").pack(anchor="w", padx=5)


    def process_queue(self):
        while not self.msg_queue.empty():
            try:
                msg = self.msg_queue.get_nowait()
                self.log_txt.configure(state='normal')
                timestamp = datetime.now().strftime('%H:%M:%S')
                if msg.startswith("=") or "📄" in msg:
                     self.log_txt.insert(tk.END, f"{msg}\n")
                else:
                    self.log_txt.insert(tk.END, f"[{timestamp}] {msg}\n")
                self.log_txt.see(tk.END)
                self.log_txt.configure(state='disabled')
            except queue.Empty:
                break
        self.root.after(100, self.process_queue)

    def save(self):
        self.config["MCP_ENDPOINT"] = self.mcp_ent.get().strip()
        k = self.key_ent.get().strip()
        if "系統" not in k: self.config["GOOGLE_API_KEY"] = k
        save_config(self.config)
        messagebox.showinfo("成功", "已保存")

    def toggle(self):
        if self.process: self.stop()
        else: self.start()

    def start(self):
        env = os.environ.copy()
        env["MCP_ENDPOINT"] = self.mcp_ent.get().strip()
        env["PYTHONUNBUFFERED"] = "1"
        si = None
        if sys.platform == 'win32':
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        self.process = subprocess.Popen(
            [sys.executable, "-u", os.path.abspath(__file__), "--pipe"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, startupinfo=si
        )
        self.btn_run.config(text="停止服務")
        self.msg_queue.put("正在啟動服務程序...")
        self.reader_thread = threading.Thread(target=self.read_log_thread, daemon=True)
        self.reader_thread.start()

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process = None
        self.btn_run.config(text="啟動服務")
        self.msg_queue.put("服務已停止")

    def read_log_thread(self):
        if not self.process: return
        for line in iter(self.process.stderr.readline, ''):
            if not line: break
            clean = line.strip()
            if "[GUI_LOG]" in clean:
                self.msg_queue.put(clean.replace("[GUI_LOG]", ""))
            elif "ERR" in clean:
                self.msg_queue.put(f"SYS: {clean}")
        if self.process:
            self.msg_queue.put("服務進程意外退出")
            self.root.after(0, self.stop)

    def on_close(self):
        self.stop()
        self.root.destroy()

if __name__ == "__main__":
    if "--server" in sys.argv: run_mcp_server()
    elif "--pipe" in sys.argv: run_mcp_pipe()
    else: MCPApp().root.mainloop()
