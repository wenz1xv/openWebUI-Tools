"""
title: DeepSeek R1 with searxng search
author: zyman
author_url: https://github.com/wenz1xv/openWebUI-Tools/
description: In OpenWebUI, displays the thought chain of the DeepSeek R1 model and searxng searchs (version 0.5.6 or higher)
version: 0.1.0  
licence: MIT 
"""

import json
import httpx
import re
from typing import AsyncGenerator, Callable, Awaitable, Any
from pydantic import BaseModel, Field
import asyncio
from typing import List, Union, Generator, Iterator
import requests
from bs4 import BeautifulSoup
import urllib.parse
import os
from requests import get
import concurrent.futures
from urllib.parse import urlparse, urljoin
import unicodedata


def _parse_response(response):

    if "data" in response:
        data = response["data"]
        if "webPages" in data:
            webPages = data["webPages"]
            if "value" in webPages:
                result = [
                    {
                        "id": item.get("id", ""),
                        "name": item.get("name", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("snippet", ""),
                        "summary": item.get("summary", ""),
                        "siteName": item.get("siteName", ""),
                        "siteIcon": item.get("siteIcon", ""),
                        "datePublished": item.get("datePublished", "")
                        or item.get("dateLastCrawled", ""),
                    }
                    for item in webPages["value"]
                ]
    return result


class Pipe:
    class Valves(BaseModel):
        DEEPSEEK_API_BASE_URL: str = Field(
            default="https://api.deepseek.com/v1",
            description="API的基础请求地址",
        )
        DEEPSEEK_API_KEY: str = Field(default="", description="API密钥，可从控制台获取")
        DEEPSEEK_API_MODEL: str = Field(
            default="deepseek-r1",
            description="API请求的模型名称，默认为 deepseek-r1",
        )
        SEARXNG_ENGINE_API_BASE_URL: str = Field(
            default="https://example.com/search",
            description="Searxng地址",
        )
        IGNORED_WEBSITES: str = Field(
            default="",
            description="忽略的搜索网站",
        )
        RETURNED_SCRAPPED_PAGES_NO: int = Field(
            default=3,
            description="处理的网页数量",
        )
        SCRAPPED_PAGES_NO: int = Field(
            default=5,
            description="搜索的网页数量",
        )
        PAGE_CONTENT_WORDS_LIMIT: int = Field(
            default=5000,
            description="网页内容长度限制",
        )
        CITATION_LINKS: bool = Field(
            default=False,
            description="是否在回答中引用网页",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.data_prefix = "data:"
        self.emitter = None
        self.type = "manifold"
        self.id = "engine_search"
        self.name = "engines/"
        self.search_result = ""

    def pipes(self):
        return [
            {
                "id": self.valves.DEEPSEEK_API_MODEL,
                "name": self.valves.DEEPSEEK_API_MODEL,
            }
        ]

    async def pipe(
        self,
        body: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
        results=None,
    ) -> AsyncGenerator[str, None]:
        """主处理管道（已移除缓冲）"""
        user_input = self._extract_user_input(body)

        if not user_input:
            yield json.dumps({"error": "No search query provided"}, ensure_ascii=False)
            return

        model = body.get("model", "")
        print(f"Received model: {model}")  # Debug print

        if isinstance(results, str):
            try:
                results = int(results)
            except ValueError:
                yield json.dumps(
                    {"error": "Invalid number of results '{results}'"},
                    ensure_ascii=False,
                )
                return

        search_results = self._search_searxng(user_input, results)
        urls = "\n".join([result["url"] for result in search_results])
        yield f"""
        <details>
            <summary>正在阅读 {user_input} 搜索结果 </summary>
            {urls}
        </details>
        """
        await asyncio.sleep(0.1)
        self.search_result = await self._get_result(search_results)

        thinking_state = {"thinking": -1}  # 使用字典来存储thinking状态
        self.emitter = __event_emitter__

        # 验证配置
        if not self.valves.DEEPSEEK_API_KEY:
            yield json.dumps({"error": "未配置API密钥"}, ensure_ascii=False)
            return

        # 获取用户输入
        messages = body["messages"]
        user_input = messages[-1]["content"]

        # 合并搜索结果和用户输入
        combined_input = (
            f"Search Results: {self.search_result}\n\nUser Input: {user_input}"
        )
        # 准备请求参数
        headers = {
            "Authorization": f"Bearer {self.valves.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }

        try:
            # 模型ID提取
            model_id = body["model"].split(".", 1)[-1]
            payload = {**body, "model": model_id}

            # 将合并后的输入替换掉原来的用户输入部分
            payload["messages"] = [{"role": "user", "content": combined_input}]
            # 处理消息以防止连续的相同角色
            messages = payload["messages"]
            i = 0
            while i < len(messages) - 1:
                if messages[i]["role"] == messages[i + 1]["role"]:
                    # 插入具有替代角色的占位符消息
                    alternate_role = (
                        "assistant" if messages[i]["role"] == "user" else "user"
                    )
                    messages.insert(
                        i + 1,
                        {"role": alternate_role, "content": "[Unfinished thinking]"},
                    )
                i += 1

            # yield json.dumps(payload, ensure_ascii=False)

            # 发起API请求
            async with httpx.AsyncClient(http2=True) as client:
                async with client.stream(
                    "POST",
                    f"{self.valves.DEEPSEEK_API_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=300,
                ) as response:
                    # 错误处理
                    if response.status_code != 200:
                        error = await response.aread()
                        yield self._format_error(response.status_code, error)
                        return

                    # 流式处理响应
                    async for line in response.aiter_lines():
                        if not line.startswith(self.data_prefix):
                            continue

                        # 截取 JSON 字符串
                        json_str = line[len(self.data_prefix) :]

                        # 去除首尾空格后检查是否为结束标记
                        if json_str.strip() == "[DONE]":
                            return

                        try:
                            data = json.loads(json_str)
                        except json.JSONDecodeError as e:
                            # 格式化错误信息，这里传入错误类型和详细原因（包括出错内容和异常信息）
                            error_detail = f"解析失败 - 内容：{json_str}，原因：{e}"
                            yield self._format_error("JSONDecodeError", error_detail)
                            return

                        choice = data.get("choices", [{}])[0]

                        # 结束条件判断
                        if choice.get("finish_reason"):
                            return

                        # 状态机处理
                        state_output = await self._update_thinking_state(
                            choice.get("delta", {}), thinking_state
                        )
                        if state_output:
                            yield state_output  # 直接发送状态标记
                            if state_output == "<think>":
                                yield "\n"

                        # 内容处理并立即发送
                        content = self._process_content(choice["delta"])
                        if content:
                            if content.startswith("<think>"):
                                match = re.match(r"^<think>", content)
                                if match:
                                    content = re.sub(r"^<think>", "", content)
                                    yield "<think>"
                                    await asyncio.sleep(0.1)
                                    yield "\n"

                            elif content.startswith("</think>"):
                                match = re.match(r"^</think>", content)
                                if match:
                                    content = re.sub(r"^</think>", "", content)
                                    yield "</think>"
                                    await asyncio.sleep(0.1)
                                    yield "\n"
                            yield content

        except Exception as e:
            yield self._format_exception(e)

    def _extract_user_input(self, body: dict) -> str:
        messages = body.get("messages", [])
        if messages:
            last_message = messages[-1]
            if isinstance(last_message.get("content"), list):
                for item in last_message["content"]:
                    if item["type"] == "text":
                        return item["text"]
            else:
                return last_message.get("content", "")
        return ""

    def _search_searxng(self, query: str, results=None) -> str:
        search_engine_url = self.valves.SEARXNG_ENGINE_API_BASE_URL
        # Ensure RETURNED_SCRAPPED_PAGES_NO does not exceed SCRAPPED_PAGES_NO
        if self.valves.RETURNED_SCRAPPED_PAGES_NO > self.valves.SCRAPPED_PAGES_NO:
            self.valves.RETURNED_SCRAPPED_PAGES_NO = self.valves.SCRAPPED_PAGES_NO
        params = {
            "q": query,
            "format": "json",
            "number_of_results": self.valves.RETURNED_SCRAPPED_PAGES_NO,
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        try:
            print("Sending request to search engine")
            resp = requests.get(
                search_engine_url, params=params, headers=headers, timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            limited_results = results[: self.valves.SCRAPPED_PAGES_NO]
            print(f"Retrieved {len(limited_results)} search results")
        except requests.exceptions.RequestException as e:
            print(f"Error during search: {str(e)}")
            return f"An error occurred while searching searxng: {str(e)}"
        return limited_results

    async def _get_result(self, limited_results: list) -> str:
        functions = HelpFunctions()
        results_json = []
        if limited_results:
            print(f"Processing search results")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        functions.process_search_result, result, self.valves
                    )
                    for result in limited_results
                ]
                for future in concurrent.futures.as_completed(futures):
                    result_json = future.result()
                    if result_json:
                        try:
                            json.dumps(result_json)
                            results_json.append(result_json)
                        except (TypeError, ValueError):
                            continue
                    if len(results_json) >= self.valves.RETURNED_SCRAPPED_PAGES_NO:
                        break
            results_json = results_json[: self.valves.RETURNED_SCRAPPED_PAGES_NO]
            formatted_results = "Searxng Search Results:\n\n"
            for i, result in enumerate(results_json):
                line = f'Website {i}: {result["title"]}\n{result["url"]}\n{result["content"]}\n{result["snippet"]}\n\n'
                formatted_results += line
            return formatted_results
        else:
            return f"Inform users that search results are empty and answer user questions with caution"

    async def _update_thinking_state(self, delta: dict, thinking_state: dict) -> str:
        """更新思考状态机（简化版）"""
        state_output = ""

        # 状态转换：未开始 -> 思考中
        if thinking_state["thinking"] == -1 and delta.get("reasoning_content"):
            thinking_state["thinking"] = 0
            state_output = "<think>"

        # 状态转换：思考中 -> 已回答
        elif (
            thinking_state["thinking"] == 0
            and not delta.get("reasoning_content")
            and delta.get("content")
        ):
            thinking_state["thinking"] = 1
            state_output = "\n</think>\n\n"

        return state_output

    def _process_content(self, delta: dict) -> str:
        """直接返回处理后的内容"""
        return delta.get("reasoning_content", "") or delta.get("content", "")

    def _format_error(self, status_code: int, error: bytes) -> str:
        # 如果 error 已经是字符串，则无需 decode
        if isinstance(error, str):
            error_str = error
        else:
            error_str = error.decode(errors="ignore")

        try:
            err_msg = json.loads(error_str).get("message", error_str)[:200]
        except Exception as e:
            err_msg = error_str[:200]
        return json.dumps(
            {"error": f"HTTP {status_code}: {err_msg}"}, ensure_ascii=False
        )

    def _format_exception(self, e: Exception) -> str:
        """异常格式化保持不变"""
        err_type = type(e).__name__
        return json.dumps({"error": f"{err_type}: {str(e)}"}, ensure_ascii=False)


class HelpFunctions:
    def __init__(self):
        pass

    def get_base_url(self, url):
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        return base_url

    def generate_excerpt(self, content, max_length=200):
        return content[:max_length] + "..." if len(content) > max_length else content

    def format_text(self, original_text):
        soup = BeautifulSoup(original_text, "html.parser")
        formatted_text = soup.get_text(separator=" ", strip=True)
        formatted_text = unicodedata.normalize("NFKC", formatted_text)
        formatted_text = re.sub(r"\s+", " ", formatted_text)
        formatted_text = formatted_text.strip()
        formatted_text = self.remove_emojis(formatted_text)
        return formatted_text

    def remove_emojis(self, text):
        return "".join(c for c in text if not unicodedata.category(c).startswith("So"))

    def process_search_result(self, result, valves):
        title_site = self.remove_emojis(result["title"])
        url_site = result["url"]
        snippet = result.get("content", "")

        # Check if the website is in the ignored list, but only if IGNORED_WEBSITES is not empty
        if valves.IGNORED_WEBSITES:
            base_url = self.get_base_url(url_site)
            if any(
                ignored_site.strip() in base_url
                for ignored_site in valves.IGNORED_WEBSITES.split(",")
            ):
                return None

        try:
            response_site = requests.get(url_site, timeout=20)
            if response_site.headers.get("Content-Type", "").find("pdf") > -1:
                print(f"{url_site} is pdf")
                return None
            response_site.raise_for_status()
            html_content = response_site.text

            soup = BeautifulSoup(html_content, "html.parser")
            content_site = self.format_text(soup.get_text(separator=" ", strip=True))

            truncated_content = self.truncate_to_n_words(
                content_site, valves.PAGE_CONTENT_WORDS_LIMIT
            )

            return {
                "title": title_site,
                "url": url_site,
                "content": truncated_content,
                "snippet": self.remove_emojis(snippet),
            }

        except requests.exceptions.RequestException as e:
            return None

    def truncate_to_n_words(self, text, token_limit):
        tokens = text.split()
        truncated_tokens = tokens[:token_limit]
        return " ".join(truncated_tokens)
