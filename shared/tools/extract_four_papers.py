# -*- coding: utf-8 -*-
"""Extract the four paper PDFs to text for anchor mining (per-page fault tolerant)."""
from pathlib import Path
import sys

import fitz

PAPERS = Path(r".\20260715_BrainGate_After_T11\papers")
OUT = Path(r".\shared\result\papers_txt")
OUT.mkdir(parents=True, exist_ok=True)

PDFS = [
    ("willett2023", PAPERS / "2023" / "2023-08-31 - Willett et al. - A high-performance speech neuroprosthesis.pdf"),
    ("card2024", PAPERS / "2024" / "2024-08-15 - Card et al. - An accurate and rapidly calibrating speech neuroprosthesis.pdf"),
    ("kunz2025", PAPERS / "2025" / "2025-08-21 - Kunz et al. - Inner speech in motor cortex and implications for speech neuroprostheses.pdf"),
    ("jude2026", PAPERS / "2026" / "2026-04-28 - Jude et al. - Decoding intended speech with an intracortical brain-computer interface in a person with long-standing anarthria and locked-in syndrome.pdf"),
]
want = set(sys.argv[1:]) or {n for n, _ in PDFS}

for name, pdf in PDFS:
    if name not in want:
        continue
    doc = fitz.open(str(pdf))
    target = OUT / f"{name}.txt"
    with target.open("w", encoding="utf-8", newline="") as h:
        for i, page in enumerate(doc, 1):
            h.write(f"\n\n===== PAGE {i} OF {doc.page_count} =====\n\n")
            h.write(page.get_text())
    print(name, doc.page_count, "pages ->", target.name)
