# ShopPilot 当前可用商品数据源

## 1. 当前结论

项目已移除无法获得有效凭证的 eBay Browse API 和 Rakuten Ichiba API 运行路径。当前保留两类可稳定复现的数据源：

1. `synthetic_hybrid`：1,200 条离线合成商品，用于六个基准品类的搜索、排序、评测和跨平台比较；
2. `public_demo_catalog_snapshot`：1,000 条多来源公开测试、模拟商城和开放价格数据，用于补充综合品类和验证数据采集链路。

Amazon、Shopee、AliExpress、eBay 仍作为合成数据中的平台分区存在，但不代表已接入这些平台的实时官方商品。

## 2. Provider 架构

```text
item_search
    ↓
CatalogSearchRequest
    ↓
Catalog Router
    ├── public_demo → public_demo_catalog_snapshot
    └── 离线平台分区 → synthetic_hybrid
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
SHOPPILOT_DATASET_DIR=./data/merged_catalog
SHOPPILOT_DATASET_SCHEMA_VERSION=2
```

用途：

- BM25、BGE、Faiss、RRF 和 LTR 离线评测；
- 预算、材质等硬约束过滤；
- 四个平台分区的离线跨平台比较；
- Agent、Checkpoint、记忆、可靠性和前端演示；
- 可复现自动化测试。

必须标注：

```text
data_origin=synthetic
verification_status=synthetic
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

> 我为商品数据设计了统一 Catalog Provider 层。项目默认使用 1,200 条可复现的离线合成商品，用 Amazon、Shopee、AliExpress、eBay 四个大电商平台风格分区来做检索评测和面试演示；另有 1,000 条可选的模拟电商目录用于补充综合品类。构建器会做统一 Schema、去重、分类映射、来源与数据边界标注以及品类占比控制，再进入检索、比价和 Agent 编排流程。

不要表述为：

- 已接入 eBay、Rakuten、Amazon 等实时官方商品；
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
