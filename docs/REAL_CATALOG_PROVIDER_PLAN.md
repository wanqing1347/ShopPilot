# ShopPilot 当前可用商品数据源

## 1. 当前结论

项目当前使用本地离线快照与可选 public_demo 目录两条路径。离线快照用于稳定复现，public_demo 只在显式开启后调用：

1. `offline_snapshot`：6,616 条已归一化缓存观察，映射到 Amazon、Walmart、eBay 三个平台回退分区，用于检索、排序、评测和跨平台比较；
2. `public_demo_catalog_snapshot`：1,000 条多来源公开测试、模拟商城和开放价格数据，用于补充综合品类和验证数据采集链路；

离线快照统一映射到 Amazon、Walmart、eBay 三个平台离线分区。它们是历史或公开来源缓存观察，不代表实时官方商品、库存或结算价格。

## 2. Provider 架构

```text
item_search
    ↓
CatalogSearchRequest
    ↓
Catalog Router
    ├── public_demo → public_demo_catalog_snapshot
    └── 三平台离线分区 → offline_snapshot
    ↓
统一 Candidate
    ↓
price_compare → shipping_calc → item_picker → shopping_summary
```

```text
app/catalog/
├── base.py
├── public_demo_taxonomy.py
├── router.py
└── providers/
    ├── public_demo.py
    └── synthetic.py
```

## 3. 合成数据

```env
SHOPPILOT_DATASET_DIR=./data/offline_catalog
SHOPPILOT_DATASET_SCHEMA_VERSION=2
```

用途：

- BM25、BGE、Faiss、RRF 和 LTR 离线评测；
- 预算、材质等硬约束过滤；
- Amazon/Walmart/eBay 三平台分区的离线跨平台比较；
- Agent、Checkpoint、记忆、可靠性和前端演示；
- 可复现自动化测试。

必须标注：

```text
data_origin=offline_snapshot
verification_status=cached
```

## 4. 公开测试商品目录

```env
SHOPPILOT_PUBLIC_DEMO_ENABLED=false
SHOPPILOT_PUBLIC_DEMO_DATA_FILE=./data/public_demo/products.jsonl
```

重新构建：

```powershell
.venv312\Scripts\python.exe scripts\build_public_demo_catalog.py --target 1000
```

当前包含 11 个来源：

- DummyJSON Products；
- Platzi Fake Store API；
- web-scraping.dev；
- Automation Exercise Products API；
- Practice Software Testing Products API；
- FakeStoreAPI；
- Shopify Mock.shop；
- Vendure Read-only Demo；
- Open Food Facts Open Prices；
- ScrapeMe；
- Books to Scrape。

来源类型包括：

```text
fake_api
mock_store_graphql_api
read_only_demo_graphql_api
automation_practice_api
software_testing_demo_api
mock_commerce_fixture
mock_book_catalog
open_price_observation
```

构建脚本统一：

- 商品 ID 与来源 URL；
- 多币种价格及人民币演示换算；
- `category_key`；
- 中英文分类别名和标签；
- 图书题材及中文题材名；
- 数据来源、许可证和真实性状态；
- 按 ID/URL 去重；
- 统一品类占比硬上限。

来源标记：

```text
platform=public_demo
data_origin=public_demo_catalog
verification_status=public_demo
provider=<具体测试或开放数据来源>
```

运行时 Provider：

```text
public_demo_catalog_snapshot
```

## 5. 当前目录规模与分布

2026-08-06 快照：

```text
1000 条商品
11 个来源
30 个统一品类键
220 个来源子品类
```

主要分布：

```text
食品杂货      200
收藏玩具      186
图书          156
服饰           92
电子产品       66
其他商品       54
鞋靴           43
厨房用品       29
运动配件       25
家具           24
五金工具       20
智能手机       19
```

任一统一品类最多约占目标数量的 20%，避免单一大数据源主导整个目录。

Open Prices 数据属于开放的历史价格观察，不能据此声称商品当前有库存或价格仍有效。

## 6. 合规和可复现原则

- 不直接接入或访问 Amazon、淘宝、京东等未明确授权的商业站点；
- 不绕过登录、验证码、访问限制或反爬措施；
- 公开测试/Mock 目录在运行时读取并遵守 `robots.txt`；
- 对公开测试/Mock 目录请求主动限速；
- Open Prices 使用自定义 User-Agent，并只读取少量分页；
- 数据来源政策写入 `snapshot_summary.json`；
- 所有结果明确标注为测试、模拟或开放数据。

## 7. 面试表述

推荐表述：

> 我为商品数据设计了统一 Catalog Provider 层。项目默认使用 6,616 条离线缓存观察，并映射到 Amazon、Walmart、eBay 三个平台离线分区。构建器会做统一 Schema、去重、分类映射和数据边界标注，再进入检索、比价和 Agent 编排流程。

不要表述为：

- 离线快照等于当前官方库存或实时价格；
- 已接入所有平台的实时官方商品；
- 能获取真实库存、成交量或最终结算价；
- 公开测试目录中的商品都可以实际购买；
- Open Prices 历史观察等于当前零售价；
- Fake API 数据属于真实商家授权数据。

## 8. 后续扩展原则

未来只有在满足以下条件后，才重新添加真实平台 Provider：

- 已获得合法、有效并可持续使用的 API 凭证；
- 可以完成真实接口冒烟和自动化测试；
- 明确平台条款、配额、缓存和展示要求；
- 商品来源、新鲜度与授权状态可以被准确标记。
