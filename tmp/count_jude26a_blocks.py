"""One-off: scan all Jude 2026a source-data workbooks for block inventories.

Counts unique (session, block) tokens matching the paper's
``tNN.YYYY.MM.DD[-PU]-Block-N`` ID scheme in every sheet of every workbook,
to locate which (if any) sheet accounts for the hand-filled "46/98" block
counts in the benchmark YAML span notes.
"""
import io
import re
import zipfile
from pathlib import Path

import openpyxl

ZPATH = Path("tmp/jude26a_sourcedata/moesm7.zip")
PAT = re.compile(r"(t\d+\.\d{4}\.\d{2}\.\d{2})(?:PU)?-Block-(\d+)")

z = zipfile.ZipFile(ZPATH)
names = [n for n in z.namelist() if n.lower().endswith(".xlsx") and "__MACOSX" not in n]
for n in sorted(names):
    book = openpyxl.load_workbook(io.BytesIO(z.read(n)), read_only=True, data_only=True)
    for sheet in book.sheetnames:
        ws = book[sheet]
        blocks, pu_blocks, sessions = set(), set(), set()
        rows = 0
        for row in ws.values:
            rows += 1
            for cell in row:
                if cell is None:
                    continue
                m = PAT.search(str(cell))
                if m:
                    blocks.add((m.group(1), m.group(2)))
                    if "PU" in str(cell):
                        pu_blocks.add((m.group(1), m.group(2)))
                    sessions.add(m.group(1))
        if blocks:
            t17 = sorted(b for b in blocks if b[0].startswith("t17"))
            t18 = sorted(b for b in blocks if b[0].startswith("t18"))
            print(f"{Path(n).name} | {sheet} | rows={rows}")
            print(f"    T17: {len(t17)} blocks over {len({b[0] for b in t17})} sessions"
                  f" (PU: {len([b for b in t17 if b in pu_blocks])})")
            print(f"    T18: {len(t18)} blocks over {len({b[0] for b in t18})} sessions"
                  f" (PU: {len([b for b in t18 if b in pu_blocks])})")
    book.close()
