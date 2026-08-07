from __future__ import annotations

import re

CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "travel_storage": ("travel storage", "packing cube", "travel pouch", "旅行收纳", "收纳袋", "收纳包"),
    "backpack": ("backpack", "rucksack", "背包", "双肩包", "书包"),
    "keyboard": ("keyboard", "mechanical keyboard", "键盘", "机械键盘"),
    "headphones": ("headphone", "headphones", "earbud", "earbuds", "airpods", "headset", "耳机", "耳麦", "蓝牙耳机"),
    "thermos": ("thermos", "vacuum bottle", "tumbler", "保温杯", "保温瓶", "随行杯"),
    "coffee_cup": ("coffee cup", "coffee mug", "mug", "咖啡杯", "马克杯", "杯子"),
    "books": ("book", "books", "novel", "novels", "reading", "literature", "图书", "书籍", "小说", "读物", "文学"),
    "beauty": ("beauty", "makeup", "cosmetics", "mascara", "lipstick", "美妆", "彩妆", "化妆品", "睫毛膏", "口红"),
    "fragrances": ("fragrance", "fragrances", "perfume", "cologne", "香水", "香氛"),
    "furniture": ("furniture", "chair", "table", "bed", "sofa", "家具", "椅子", "桌子", "床", "沙发"),
    "groceries": ("grocery", "groceries", "food", "drink", "snack", "食品", "饮料", "零食", "杂货"),
    "home_decoration": ("home decoration", "home decor", "decoration", "家居装饰", "装饰品", "摆件"),
    "kitchen_accessories": ("kitchen accessory", "kitchen accessories", "cookware", "餐厨", "厨房用品", "厨具"),
    "laptops": ("laptop", "laptops", "notebook computer", "笔记本电脑", "笔记本"),
    "apparel": ("apparel", "clothes", "clothing", "shirt", "dress", "hoodie", "服饰", "衣服", "衬衫", "连衣裙", "卫衣"),
    "footwear": ("footwear", "shoe", "shoes", "boot", "boots", "sandal", "sneaker", "鞋", "鞋子", "运动鞋", "靴子", "凉鞋"),
    "watches": ("watch", "watches", "wristwatch", "手表", "腕表"),
    "mobile_accessories": ("mobile accessory", "phone accessory", "charger", "power bank", "手机配件", "充电器", "充电宝"),
    "motorcycle": ("motorcycle", "motorbike", "helmet", "摩托车", "摩托", "头盔"),
    "skincare": ("skin care", "skincare", "cream", "serum", "护肤", "护肤品", "面霜", "精华"),
    "smartphones": ("smartphone", "smartphones", "mobile phone", "iphone", "android phone", "手机", "智能手机"),
    "sports_accessories": ("sports accessory", "sports accessories", "fitness", "运动配件", "健身用品", "体育用品"),
    "sunglasses": ("sunglasses", "sun glasses", "墨镜", "太阳镜"),
    "tablets": ("tablet", "tablets", "ipad", "平板电脑", "平板"),
    "vehicles": ("vehicle", "vehicles", "car", "automobile", "汽车", "车辆"),
    "bags": ("bag", "bags", "handbag", "purse", "包", "手提包", "女包"),
    "jewellery": ("jewellery", "jewelry", "necklace", "ring", "earring", "珠宝", "首饰", "项链", "戒指", "耳环"),
    "electronics": ("electronics", "electronic", "gadget", "电子产品", "数码产品", "电器"),
    "lighting": ("lighting", "lamp", "light", "灯具", "台灯", "照明"),
    "tools": (
        "tool",
        "tools",
        "hand tool",
        "power tool",
        "hammer",
        "pliers",
        "screwdriver",
        "drill",
        "saw",
        "wrench",
        "工具",
        "五金工具",
        "手动工具",
        "电动工具",
        "锤子",
        "钳子",
        "螺丝刀",
        "电钻",
    ),
    "collectibles": (
        "collectible",
        "collectibles",
        "toy",
        "toys",
        "pokemon",
        "pokémon",
        "figure",
        "trading card",
        "收藏品",
        "玩具",
        "手办",
        "宝可梦",
    ),
    "miscellaneous": ("miscellaneous", "misc", "other", "其他", "杂项"),
}

CATEGORY_LABEL_ZH: dict[str, str] = {
    "travel_storage": "旅行收纳",
    "backpack": "背包",
    "keyboard": "键盘",
    "headphones": "耳机",
    "thermos": "保温杯",
    "coffee_cup": "咖啡杯",
    "books": "图书",
    "beauty": "美妆",
    "fragrances": "香水",
    "furniture": "家具",
    "groceries": "食品杂货",
    "home_decoration": "家居装饰",
    "kitchen_accessories": "厨房用品",
    "laptops": "笔记本电脑",
    "apparel": "服饰",
    "footwear": "鞋靴",
    "watches": "手表",
    "mobile_accessories": "手机配件",
    "motorcycle": "摩托车用品",
    "skincare": "护肤",
    "smartphones": "智能手机",
    "sports_accessories": "运动配件",
    "sunglasses": "太阳镜",
    "tablets": "平板电脑",
    "vehicles": "汽车",
    "bags": "箱包",
    "jewellery": "珠宝首饰",
    "electronics": "电子产品",
    "lighting": "灯具",
    "tools": "五金工具",
    "collectibles": "收藏玩具",
    "miscellaneous": "其他商品",
}

SOURCE_CATEGORY_KEY: dict[str, str] = {
    "beauty": "beauty",
    "fragrances": "fragrances",
    "furniture": "furniture",
    "groceries": "groceries",
    "home-decoration": "home_decoration",
    "kitchen-accessories": "kitchen_accessories",
    "laptops": "laptops",
    "mens-shirts": "apparel",
    "mens-shoes": "footwear",
    "mens-watches": "watches",
    "mobile-accessories": "mobile_accessories",
    "motorcycle": "motorcycle",
    "skin-care": "skincare",
    "smartphones": "smartphones",
    "sports-accessories": "sports_accessories",
    "sunglasses": "sunglasses",
    "tablets": "tablets",
    "tops": "apparel",
    "vehicle": "vehicles",
    "womens-bags": "bags",
    "womens-dresses": "apparel",
    "womens-jewellery": "jewellery",
    "womens-shoes": "footwear",
    "womens-watches": "watches",
    "clothes": "apparel",
    "electronics": "electronics",
    "shoes": "footwear",
    "miscellaneous": "miscellaneous",
    "apparel": "apparel",
    "footwear": "footwear",
    "consumables": "groceries",
    "men's clothing": "apparel",
    "women's clothing": "apparel",
    "jewelery": "jewellery",
    "tops": "apparel",
    "tshirts": "apparel",
    "dress": "apparel",
    "dresses": "apparel",
    "kids": "apparel",
    "pliers": "tools",
    "hand tools": "tools",
    "power tools": "tools",
    "other": "tools",
    "screwdrivers": "tools",
    "hammers": "tools",
    "saws": "tools",
    "wrenches": "tools",
    "drills": "tools",
    "pokemon": "collectibles",
    "collectibles": "collectibles",
    "toys": "collectibles",
    "computers": "electronics",
    "outdoor equipment": "sports_accessories",
    "sports & outdoor": "sports_accessories",
} 

BOOK_GENRE_ZH: dict[str, str] = {
    "travel": "旅行",
    "mystery": "悬疑",
    "historical fiction": "历史小说",
    "sequential art": "漫画",
    "classics": "经典文学",
    "philosophy": "哲学",
    "romance": "爱情小说",
    "womens fiction": "女性小说",
    "fiction": "小说",
    "childrens": "儿童读物",
    "religion": "宗教",
    "nonfiction": "非虚构",
    "music": "音乐",
    "science fiction": "科幻",
    "sports and games": "体育与游戏",
    "fantasy": "奇幻",
    "new adult": "新成人",
    "young adult": "青少年",
    "science": "科学",
    "poetry": "诗歌",
    "paranormal": "超自然",
    "art": "艺术",
    "psychology": "心理学",
    "autobiography": "自传",
    "parenting": "育儿",
    "adult fiction": "成人小说",
    "humor": "幽默",
    "horror": "恐怖",
    "history": "历史",
    "food and drink": "美食饮品",
    "christian fiction": "基督教小说",
    "business": "商业",
    "biography": "传记",
    "thriller": "惊悚",
    "contemporary": "当代文学",
    "spirituality": "灵性",
    "academic": "学术",
    "self help": "自助成长",
    "historical": "历史",
    "christian": "基督教",
    "suspense": "悬念",
    "short stories": "短篇小说",
    "novels": "长篇小说",
    "health": "健康",
    "politics": "政治",
    "cultural": "文化",
    "erotica": "成人文学",
    "crime": "犯罪",
    "default": "综合图书",
    "add a comment": "综合图书",
}

_TITLE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(keyboard|keycaps?)\b", re.I), "keyboard"),
    (re.compile(r"\b(headphones?|earbuds?|airpods?|headset)\b", re.I), "headphones"),
    (re.compile(r"\b(backpack|rucksack)\b", re.I), "backpack"),
    (re.compile(r"\b(thermos|vacuum bottle|tumbler)\b", re.I), "thermos"),
    (re.compile(r"\b(coffee mug|coffee cup|mug)\b", re.I), "coffee_cup"),
    (re.compile(r"\b(packing cube|travel pouch|travel organizer)\b", re.I), "travel_storage"),
    (re.compile(r"\b(laptop|notebook computer|macbook)\b", re.I), "laptops"),
    (re.compile(r"\b(tablet|ipad)\b", re.I), "tablets"),
    (re.compile(r"\b(smartphone|iphone|android phone)\b", re.I), "smartphones"),
    (re.compile(r"\b(shoes?|boots?|sandals?|sneakers?|slides?)\b", re.I), "footwear"),
    (re.compile(r"\b(shirts?|t-?shirts?|dress|dresses|hoodies?|sweatpants|shorts)\b", re.I), "apparel"),
    (
        re.compile(r"\b(hammer|pliers?|screwdrivers?|drill|saw|wrench|chisel|tool set)\b", re.I),
        "tools",
    ),
    (re.compile(r"\b(pokemon|pokémon|collectible|figurine|trading card|toy)\b", re.I), "collectibles"),
)


def infer_category_key(source_category: str, title: str = "") -> str:
    for pattern, key in _TITLE_RULES:
        if pattern.search(title):
            return key
    normalized = " ".join(source_category.strip().lower().replace("_", "-").split())
    if normalized in SOURCE_CATEGORY_KEY:
        return SOURCE_CATEGORY_KEY[normalized]
    spaced = normalized.replace("-", " ")
    for source_key, category_key in sorted(
        SOURCE_CATEGORY_KEY.items(), key=lambda pair: len(pair[0]), reverse=True
    ):
        token = source_key.replace("-", " ")
        if token and token in spaced:
            return category_key
    return "miscellaneous"


def aliases_for(category_key: str, source_category: str | None = None) -> list[str]:
    values = list(CATEGORY_ALIASES.get(category_key, (category_key,)))
    if source_category:
        normalized = " ".join(source_category.strip().lower().replace("_", " ").replace("-", " ").split())
        values.append(source_category)
        values.append(normalized)
        if category_key == "books":
            values.extend(("图书", "书籍", "小说"))
            if translated := BOOK_GENRE_ZH.get(normalized):
                values.append(translated)
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value).split()).strip()
        lowered = cleaned.lower()
        if cleaned and lowered not in seen:
            seen.add(lowered)
            result.append(cleaned)
    return result
