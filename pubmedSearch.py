"""
title: Searching for literature and summary using pubmed based on searxng
author: zyman
author_url: https://github.com/wenz1xv/openWebUI-Tools/
version: 0.1
license: MIT
"""

import asyncio
import aiohttp
from typing import Callable, Any, Optional, List
from urllib.parse import quote
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

EmitterType = Optional[Callable[[dict], Any]]


class EventEmitter:
    def __init__(self, event_emitter: EmitterType):
        self.event_emitter = event_emitter

    async def emit(self, event_type: str, data: dict):
        if self.event_emitter:
            await self.event_emitter({"type": event_type, "data": data})

    async def update_status(
        self, description: str, done: bool, action: str, urls: List[str]
    ):
        await self.emit(
            "status",
            {"done": done, "action": action, "description": description, "urls": urls},
        )

    async def send_citation(self, title: str, url: str, content: str):
        await self.emit(
            "citation",
            {
                "document": [content],
                "metadata": [{"name": title, "source": url, "html": False}],
            },
        )


class Tools:
    def __init__(self):
        self.valves = self.Valves()

    class Valves(BaseModel):
        SEARXNG_URL: str = Field(
            default="https://example.com/search",
            description="The base URL for Search Engine",
        )
        PAGES_NO: int = Field(
            default=3,
            description="检索搜索文献数量/the number of literature searched",
        )
        PAGE_CONTENT_WORDS_LIMIT: int = Field(
            default=2000,
            description="网页内容截断长度/Web Content Truncation Length",
        )
        SUMMARY: bool = Field(
            default=False,
            description="启用API总结网页，可以使AI阅读更多文献/Enabling API summarization pages enables AI to read more literature",
        )
        API_URL: str = Field(
            default="https://dashscope.aliyuncs.com/compatible-mode/v1",
            description="网页总结API，默认阿里百炼/Web Summary API, Default Ali",
        )
        API_KEY: str = Field(default="", description="API密钥")
        API_MODEL: str = Field(
            default="qwen-long",
            description="API请求的模型名称，默认为 qwen-long/Model name of the API request, default is qwen-long",
        )

    async def modify_query(self, query: str) -> str:
        async with AsyncOpenAI(
            api_key=self.valves.API_KEY,
            base_url=self.valves.API_URL,
        ) as client:
            response = await client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Generate a English search term. No other text. Based on: {query}.",
                    }
                ],
                model=self.valves.API_MODEL,
                stream=False,
            )
        query1 = response.to_dict()["choices"][0]["message"]["content"]
        return query1

    async def scholar_search(
        self, query: str, user_request: str, __event_emitter__: EmitterType = None
    ) -> str:
        """
        Searches scholarly literature based on query content and returns links and abstracts.
        This function is called only if a Search request for literature is explicitly made.

        :param query: The search query
        :param user_request: The user's original request or query
        :param __event_emitter__: Optional event emitter for status updates
        :return: Combined results from search and web scraping
        """
        emitter = EventEmitter(__event_emitter__)

        if self.valves.SEARXNG_URL == "https://example.com/search":
            await emitter.update_status(
                "请设置SEARXNG搜索地址！/Please set the SEARXNG search address!", True, "web_search", []
            )
            raise ValueError("请设置SEARXNG搜索地址！/Please set the SEARXNG search address!")

        if self.valves.SUMMARY and self.valves.API_KEY == "":
            await emitter.update_status("请填写API KEY！/Please fill in the API KEY!", True, "web_search", [])
            raise ValueError("请填写API KEY！/Please fill in the API KEY!")

        await emitter.update_status(
            f"正在优化搜索关键词/Optimize Search Keywords: {query}", False, "web_search", []
        )

        query = await self.modify_query(query)

        await emitter.update_status(
            f"正在pubmed搜索/Searching: {query}", False, "web_search", []
        )
        encoded_query = quote("!pub " + query)
        search_url = f"{self.valves.SEARXNG_URL}?q={encoded_query}&format=json&categories=science"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url) as response:
                    response.raise_for_status()
                    search_results = await response.json()
            urls = [result["url"] for result in search_results.get("results", [])]
            nlit = min(self.valves.PAGES_NO, len(urls))
            if not urls:
                await emitter.update_status(
                    "搜索未返回任何结果/Search returned no results", True, "web_search", []
                )
                return "搜索未返回任何结果/Search returned no results"
            await emitter.update_status(
                f"搜索完成,正在读取 {nlit} 结果/ Done, Reading {nlit} results",
                False,
                "web_search",
                urls,
            )
            scraped_content = await self.web_scrape(
                urls[:nlit], user_request, __event_emitter__
            )

            # 构建最终返回的字符串
            final_result = f"User query about: {query}\nOriginal request: {user_request}\n\nwebsite searching result:\n{scraped_content}"

            await emitter.update_status(
                f"搜索阅读 {nlit} 文献完毕，正在总结/Searching Done, Generating",
                True,
                "web_search",
                urls[:nlit],
            )

            return final_result

        except aiohttp.ClientError as e:
            error_message = f"搜索时发生错误/Error: {str(e)}"
            await emitter.update_status(error_message, True, "web_search", [])
            return error_message

    async def web_scrape(
        self,
        urls: List[str],
        user_request: str,
        __event_emitter__: EmitterType = None,
    ) -> str:
        """
        Scrapes content from provided URLs and returns a summary.
        Call this function when URLs are provided.

        :param urls: List of URLs of the web pages to scrape.
        :param user_request: The user's original request or query.
        :param __event_emitter__: Optional event emitter for status updates.
        :return: Combined scraped contents, or error messages.
        """
        emitter = EventEmitter(__event_emitter__)
        combined_results = []

        if self.valves.SUMMARY:
            await emitter.update_status(
                f"{self.valves.API_MODEL} 正在仔细阅读 {len(urls)} 个网页, 请稍等/Reading pages, Please wait",
                False,
                "web_search",
                urls,
            )
        else:
            await emitter.update_status(
                f"正在获取 {len(urls)} 个网页, 请稍等/Reading pages, Please wait",
                False,
                "web_search",
                urls,
            )

        async def process_url(url):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://gpts.webpilot.ai/api/read",
                        headers={
                            "Content-Type": "application/json",
                            "WebPilot-Friend-UID": "0",
                        },
                        json={
                            "link": url,
                            "ur": f"get summary of the page",
                            "lp": True,
                            "rt": False,
                            "l": "en",
                        },
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()
                content = result["content"]
                if self.valves.SUMMARY:
                    async with AsyncOpenAI(
                        api_key=self.valves.API_KEY,
                        base_url=self.valves.API_URL,
                    ) as client:
                        response = await client.chat.completions.create(
                            messages=[
                                {
                                    "role": "user",
                                    "content": f"Respond to this prompt: {user_request}.\n get summary of the data: {content}",
                                }
                            ],
                            model=self.valves.API_MODEL,
                            stream=False,
                        )
                    content = response.to_dict()["choices"][0]["message"]["content"]
                # title = url
                title = result.get("title")
                await emitter.send_citation(title, url, content)
                return f"# Title: {title if len(title) else url}\n# URL: {url}\n# Content: {content}\n"

            except aiohttp.ClientError as e:
                error_message = f"读取网页 {url} 时出错/ Error: {str(e)}"
                await emitter.update_status(error_message, False, "web_scrape", [url])
                return (
                    f"# Read Failed!\n# URL: {url}\n # Error Message: {error_message}\n"
                )

        tasks = [process_url(url) for url in urls]
        results = await asyncio.gather(*tasks)
        combined_results.extend(results)

        await emitter.update_status(
            f"已完成 {len(urls)} 个网页的读取/Reading Done", True, "web_search", urls
        )

        return "\n".join(
            [
                " ".join(
                    result.split()[: self.valves.PAGE_CONTENT_WORDS_LIMIT // len(urls)]
                )
                for result in combined_results
            ]
        )
