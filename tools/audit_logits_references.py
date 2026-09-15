"""Inventory logits/consistency references, including Office/PDF deliverables."""
import csv
import re
import subprocess
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"logits|重参数化|一致性|(?:[1-9]\.\d+e-0[4-8])|PyTorch.{0,8}ONNX", re.I)


def collect():
    for name in subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf-8').split('\0'):
        p = ROOT / name
        if not p.is_file() or name.startswith('outputs/verification/'):
            continue
        if p.suffix in ('.pptx', '.docx'):
            with zipfile.ZipFile(p) as z:
                for part in z.namelist():
                    if re.fullmatch(r'ppt/(?:slides/slide|notesSlides/notesSlide)\d+\.xml|word/document.xml', part):
                        root = ET.fromstring(z.read(part))
                        text = ' '.join(e.text or '' for e in root.iter() if e.tag.endswith('}t'))
                        for match in PATTERN.finditer(text):
                            yield name, part, text[max(0, match.start()-70):match.end()+130]
        elif p.suffix == '.pdf':
            import fitz
            with fitz.open(p) as doc:
                for i, page in enumerate(doc):
                    text = page.get_text().replace('\n', ' ')
                    for match in PATTERN.finditer(text):
                        yield name, f'page {i+1}', text[max(0, match.start()-70):match.end()+130]
        elif p.suffix.lower() in ('.md', '.json', '.csv', '.log', '.txt', '.py', '.sh', '.svg', '.yaml'):
            for i, line in enumerate(p.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
                if PATTERN.search(line):
                    yield name, str(i), line.strip()


if __name__ == '__main__':
    out = ROOT / 'outputs/verification/logits_reference_inventory.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(collect())
    with out.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'location', 'reference'])
        writer.writerows(rows)
    print(f'{len(rows)} references in {len({row[0] for row in rows})} files -> {out.relative_to(ROOT)}')
