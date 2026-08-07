# 模拟电商商品目录

## 1. 用途

该目录用于在没有海外平台开发者凭证时，为 ShopPilot 提供可运行、可复现、品类较丰富的商品检索数据。

它面向本地研发、检索评测和面试演示：数据统一标记为模拟/演示数据，**不是可实际下单的实时商城库存或最终结算报价**。

## 2. 当前规模

默认快照：

```text
data/public_demo/products.jsonl
```

当前快照固定为 1,000 条，统一标记为：

```text
platform=public_demo
verification_status=public_demo
data_origin=public_demo_catalog
```

统计摘要：

```text
data/public_demo/snapshot_summary.json
```

2026-08-06 的当前快照包含：

```text
11 个模拟/公开测试来源
30 个统一 category_key
220 个来源子品类
1000 个唯一商品 ID
1000 个唯一来源 URL
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

构建器将任一统一品类限制在目标总量的约 20% 以内，避免单一品类占比过高。

## 3. 数据边界

该目录整合公开测试 API、Mock Store、只读 Demo Shop、历史开放价格观察等非交易数据，并统一转换为 ShopPilot 的 `Candidate` Schema。

所有条目只用于模拟电商检索场景：

- 不代表 Amazon、Shopee、AliExpress、eBay 或其他真实平台的实时官方商品；
- 不代表实时库存、实时价格、实时运费、实时税费或可购买状态；
- 不用于下单、支付、商家履约或合规承诺；
- 面试展示时应表述为“模拟电商商品目录”或“离线合成/公开测试数据”。

## 4. 构建 1,000 条目录

在项目根目录执行：

```powershell
.venv312\Scripts\python.exe scripts\build_public_demo_catalog.py --target 1000
```

脚本会：

1. 读取公开测试 API / Mock Store / Demo Shop 的商品样例；
2. 读取少量历史开放价格观察；
3. 按统一 Schema 转换为 `Candidate`；
4. 按商品 ID 和来源 URL 去重；
5. 应用品类占比硬上限；
6. 原子替换 `products.jsonl`；
7. 生成来源、品类、配额和数据边界摘要。

目标范围为 100～1,500：

```powershell
.venv312\Scripts\python.exe scripts\build_public_demo_catalog.py --target 800
```

## 5. 分类与检索

统一品类包括：

```text
smartphones
laptops
tablets
headphones
keyboard
electronics
mobile_accessories
apparel
footwear
bags
beauty
skincare
groceries
furniture
kitchen_accessories
tools
collectibles
books
```

每条商品还保存：

```text
source_category
category_aliases
tags
catalog_source
source_type
genre / genre_zh（图书）
```

中英文别名参与轻量 IDF 排序，因此可以搜索：

```text
智能手机
蓝牙耳机
运动鞋
五金工具
宝可梦收藏玩具
悬疑小说
巧克力食品
```

## 6. Agent 使用

主面试演示建议优先使用默认四个模拟大平台：

```text
在 amazon、shopee、aliexpress、ebay 四个平台分别搜索预算 300 元以内的咖啡杯。
```

需要综合品类时可显式指定模拟电商目录：

```text
在公开演示商城找预算 3000 元以内的智能手机。
```

Planner 会将以下表达映射为 `public_demo`：

```text
公开演示商城
演示商城
模拟电商目录
```

用户不指定平台时，Planner 默认只使用 `amazon`、`shopee`、`aliexpress`、`ebay` 四个离线合成平台分区；`public_demo` 仅在用户明确要求演示商城/模拟电商目录时加入。

## 7. 输出展示口径

推荐清单显示：

```text
模拟电商商品目录（非真实交易平台）
```

必须避免以下表述：

- 真实商城实时库存；
- 可以直接购买的实时商品；
- 实时跨境到手价；
- 商家授权商品数据；
- 来自真实平台的官方实时数据。

运费、汇率和税费仍是项目演示估算。

## 8. 验证

```powershell
.venv312\Scripts\python.exe -m pytest tests\test_public_demo_provider.py -q
.venv312\Scripts\python.exe -m pytest -q
```

测试覆盖：

- 1,000 条快照加载；
- 至少 10 个数据来源；
- 品类占比限制；
- 手机、耳机、鞋靴、工具、收藏玩具和图书检索；
- 预算硬过滤；
- 来源字段保留；
- 演示商品零运费和零关税处理。
