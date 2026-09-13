# tools/verify_imagenet_labels.py
from pathlib import Path
PROJ = Path(__file__).resolve().parents[1]               # 仓库根，禁止写死绝对路径
P = PROJ / "labels/imagenet_classes.txt"
lines = [l.rstrip("\n") for l in open(P, encoding="utf-8")]
assert len(lines) == 1000 or len(lines) == 1001, len(lines)
short = [l.split("\t")[-1].split(",")[0].strip().lower() for l in lines if l.strip()]

anchors = {0: "tench", 207: "golden retriever", 281: "tabby", 999: "toilet tissue"}
for i, want in anchors.items():
    got = short[i]
    assert want in got or got in want, f"index {i}: 期望含 {want!r}，实际 {got!r}"
    print(f"[OK] line {i:4d} = {got!r}")
# WNID 版锚点（方式 2 生成的文件才有）
wn = [l.split("\t")[0] for l in lines if l.strip()]
if wn[0].startswith("n"):
    assert wn[0] == "n01440764" and wn[999] == "n15075141", (wn[0], wn[999])
    assert sorted(wn) == wn, "WNID 列表应为升序（字母序）"
    print("[OK] WNID 升序且首末项正确")
