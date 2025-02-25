# openWebUI-Tools
zyman 研发 OpenWebUI 工具集合


| 工具 Tool | 状态 Status | 说明 Introduction |
|---|----|---|
|DeepSearch R1| 已发布 Released| 让 DeepSeek R1 显示思维链,并让任何符合 OpenAI 接口格式的模型获得SearXNG联网搜索能力，并过滤PDF。 |
|Pubmed Search Tool| 已发布 Released | 在对话中获取pubmed数据，减少虚假文献引用 |
| A股数据获取 | 待发布 in progress| 在对话中获取实时行情数据 |


## DeepSearch Function

[DeepSearchR1](https://openwebui.com/f/zyman/deepsearchr1) 可以让 DeepSeek R1 显示思维链,并让任何符合 OpenAI 接口格式的模型快速获得联网搜索能力。支持SearXNG 搜索引擎,让 AI 模型能够**免费**获取最新的网络信息。

使用方法 参考 [如何在 Open WebUI 中显示 DeepSeek-R1 的思考过程](https://hadb.me/posts/2025/display-deepseek-r1-thinking)

### 功能介绍

####  搜索加速
基于[SearXNG 搜索](https://github.com/searxng/searxng)实现联网搜索，比open webui自带的searxng调用速度快了非常多。还过滤了搜索引擎返回的PDF文件，避免乱码作为输入，消耗大量token。

#### DeepSeek R1 思维链显示
基于[在OpwenWebUI中显示DeepSeek R1模型的思维链](https://openwebui.com/f/zgccrui/deepseek_r1)开发，兼容其他模型。

### 配置参数简介

| 变量 | 说明 | 
|------------------|-------------------|
| Deepseek Api Base Url | API的基础请求地址 | 
| Deepseek Api Key | API密钥，可从控制台获取 |
| Deepseek Api Model  | API请求的模型名称，默认为 deepseek-r1
| Searxng Engine Api Base Url |  Searxng地址 | 
| Ignored Websites |  忽略的搜索网站 |
| Returned Scrapped Pages No | 处理的网页数量 |
| Scrapped Pages No |  搜索的网页数量 |
| Page Content Words Limit | 网页内容长度限制 |
|Citation Links | 是否在回答中引用网页 |


## Pubmed Search Tool

建议在使用时给AI添加设定：”你会使用工具获取文献,当工具报错时给出报错信息和建议。当咨询你的能力边界以外的问题时或你无法获取数据时,请礼貌地说明你的局限性,并尝试提供一些相关的参考信息。“

### 功能介绍

####  pubmed检索
基于[SearXNG 搜索](https://github.com/searxng/searxng)实现联网搜索pubmed文献

#### 文献总结引用
默认使用webpilot抓取文献具体内容，由于模型输入长度限制，导致最终生成可能遗漏文献。此时可以使用总结

### 配置参数简介

| 变量 | 说明 | 
|------------------|-------------------|
| Searxng Url |  Searxng地址 | 
| Pages No | 检索搜索文献数量 |
| Page Content Words Limit | 网页内容截断长度 | 
| Summary |  是否启用API总结网页，可以使AI阅读更多文献|
| Api URL | 网页总结API，默认阿里百炼 |
| Api Key | API密钥，可从控制台获取 |
| Api Model  | API请求的模型名称，默认为 qwen-long |

### 效果预览

![pubmed搜索正常运行过程](https://github.com/user-attachments/assets/3b7e976a-4654-4620-804c-e3f2c1a09342)

![点开引用后显示具体文献摘要](https://github.com/user-attachments/assets/df33a32a-bc16-41d4-8010-b50741c37c15)







