# datasets/build_pet_class_map.py
# 运行: python datasets/build_pet_class_map.py
import json
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]               # 仓库根，禁止写死绝对路径
ANN = PROJ / "data/oxford-iiit-pet/annotations"
OUT = PROJ / "labels"; OUT.mkdir(parents=True, exist_ok=True)

# 中文名与"猫/狗"标记，键为官方文件名前缀（小写，含下划线）
CN = {
    "Abyssinian": ("阿比西尼亚猫", "猫"),           "american_bulldog": ("美国斗牛犬", "狗"),
    "american_pit_bull_terrier": ("美国比特犬", "狗"), "basset_hound": ("巴吉度猎犬", "狗"),
    "beagle": ("比格犬", "狗"),                     "Bengal": ("孟加拉豹猫", "猫"),
    "Birman": ("伯尔曼猫", "猫"),                   "Bombay": ("孟买猫", "猫"),
    "boxer": ("拳师犬", "狗"),                      "British_Shorthair": ("英国短毛猫", "猫"),
    "chihuahua": ("吉娃娃", "狗"),                  "Egyptian_Mau": ("埃及猫", "猫"),
    "english_cocker_spaniel": ("英国可卡犬", "狗"),  "english_setter": ("英国雪达犬", "狗"),
    "german_shorthaired": ("德国短毛指示犬", "狗"),  "great_pyrenees": ("大白熊犬", "狗"),
    "havanese": ("哈瓦那犬", "狗"),                 "japanese_chin": ("日本狆", "狗"),
    "keeshond": ("荷兰毛狮犬", "狗"),               "leonberger": ("兰伯格犬", "狗"),
    "Maine_Coon": ("缅因猫", "猫"),                 "miniature_pinscher": ("迷你杜宾犬", "狗"),
    "newfoundland": ("纽芬兰犬", "狗"),             "Persian": ("波斯猫", "猫"),
    "pomeranian": ("博美犬", "狗"),                 "pug": ("巴哥犬", "狗"),
    "Ragdoll": ("布偶猫", "猫"),                    "Russian_Blue": ("俄罗斯蓝猫", "猫"),
    "saint_bernard": ("圣伯纳犬", "狗"),            "samoyed": ("萨摩耶犬", "狗"),
    "scottish_terrier": ("苏格兰梗", "狗"),         "shiba_inu": ("柴犬", "狗"),
    "Siamese": ("暹罗猫", "猫"),                    "Sphynx": ("斯芬克斯猫", "猫"),
    "staffordshire_bull_terrier": ("斯塔福郡斗牛梗", "狗"), "wheaten_terrier": ("软毛麦色梗", "狗"),
    "yorkshire_terrier": ("约克夏梗", "狗"),
}

cid2prefix, cid2species = {}, {}
with open(ANN / "list.txt", encoding="utf-8") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        img_id, cid, species, breed = line.split()
        cid, species = int(cid), int(species)
        p = img_id.rsplit("_", 1)[0]                 # 关键：rsplit，不是 split
        if cid in cid2prefix:
            assert cid2prefix[cid] == p, f"CLASS-ID {cid} 对应多个前缀"
        cid2prefix[cid], cid2species[cid] = p, species

assert sorted(cid2prefix) == list(range(1, 38)), f"CLASS-ID 不是 1..37: {sorted(cid2prefix)}"
# 官方注释把猫狗 breed id 范围写反了，这里用数据本身校验：species=1 是猫
n_cat = sum(1 for s in cid2species.values() if s == 1)
assert n_cat == 12, f"猫应为 12 类，实测 {n_cat}"

prefixes  = [cid2prefix[i] for i in range(1, 38)]
classes   = [" ".join(p.title() for p in pre.split("_")) for pre in prefixes]   # torchvision 口径
cn_names  = [CN[pre][0] for pre in prefixes]
is_cat    = [CN[pre][1] == "猫" for pre in prefixes]
assert all(p in CN for p in prefixes), [p for p in prefixes if p not in CN]

class_to_idx = {c: i for i, c in enumerate(classes)}     # 扁平 {类名: 0..36}，全文唯一真源
prefix_to_idx = {p: i for i, p in enumerate(prefixes)}   # 仅脚本内自查用，不落盘
species = ["cat" if c else "dog" for c in is_cat]
hf_label_names = [p.lower() for p in prefixes]           # 与 timm/oxford-iiit-pet 的 names 同序

assert len(class_to_idx) == 37
assert class_to_idx["Abyssinian"] == 0 and class_to_idx["Yorkshire Terrier"] == 36
assert prefix_to_idx["american_pit_bull_terrier"] == 2
assert species.count("cat") == 12, species.count("cat")
print(f"[OK] num_classes=37 class_to_idx[Abyssinian]=0 class_to_idx[Yorkshire Terrier]=36 "
      f"prefix_to_idx[american_pit_bull_terrier]=2 species.cat=12")

# 落盘：labels/ 下只有这两个文件，且 json 必须是扁平 {类名: 下标}（中文名/猫狗/前缀三项由本文档 3.3 表格维护，不落盘）
(OUT / "pet_class_to_idx.json").write_text(
    json.dumps(class_to_idx, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
(OUT / "pet_classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

# 物种映射单独落盘：class_idx -> "cat"/"dog"。
# 必须独立成文件，因为「前 12 类都是猫」是**错的**——官方 CLASS-ID 顺序猫狗交错
# （idx0 Abyssinian=猫, idx1 American Bulldog=狗, idx4 Beagle=狗, idx5 Bengal=猫 ...）。
# 任何按 idx<12 判猫狗的分析都会整体错位，必须查这张表。
(OUT / "pet_species.json").write_text(
    json.dumps({str(i): species[i] for i in range(37)}, ensure_ascii=True, indent=2) + "\n",
    encoding="utf-8")
(OUT / "pet_species.txt").write_text("\n".join(species) + "\n", encoding="utf-8")
print(f"[OK] pet_species.json 已写：cat={species.count('cat')} dog={species.count('dog')}")

# 与 torchvision 交叉校验（若已安装）
try:
    from torchvision.datasets import OxfordIIITPet
    tv = OxfordIIITPet(root=str(PROJ / "data"), split="trainval", download=False).class_to_idx
    assert tv == class_to_idx, "与 torchvision class_to_idx 不一致！"
    print("[OK] 与 torchvision.class_to_idx 逐项一致")
except ModuleNotFoundError:
    print("[SKIP] 未安装 torchvision，跳过交叉校验")
print("wrote", OUT / "pet_class_to_idx.json", "和", OUT / "pet_classes.txt")
