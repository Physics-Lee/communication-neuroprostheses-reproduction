import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

const stage = path.dirname(fileURLToPath(import.meta.url));
const input = 'D:/repositories/20261012_Comment_on_Recent_Progress_of_Language_BCI/latex/source_data/SourceData_Figure2.xlsx';
const rows = JSON.parse(await fs.readFile(path.join(stage, 'offline_rows.json'), 'utf8'));
const original = JSON.parse(await fs.readFile(path.join(stage, 'original_workbook_values.json'), 'utf8'));
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(input));
const f = workbook.worksheets.getItem('f_repro_fullWER');
const before = await workbook.render({ sheetName: 'f_repro_fullWER', range: 'C1:I23', scale: 1 });
await fs.writeFile(path.join(stage, 'source_before.png'), new Uint8Array(await before.arrayBuffer()));

for (const [name, values] of Object.entries(original)) {
  const sheet = workbook.worksheets.getItem(name);
  for (const [r, row] of values.entries()) {
    for (const [c, value] of row.entries()) {
      if (value === null) sheet.getCell(r, c).values = [[null]];
    }
  }
}
for (const [index, item] of rows.entries()) {
  const row = 2 + 2 * index;
  f.getRange(`E${row}:G${row}`).values = [[item.ours, item.ours_sd, item.ours_sd]];
  f.getRange(`E${row}:G${row}`).setNumberFormat('0.00');
  f.getRange(`E${row}:G${row}`).format.horizontalAlignment = 'right';
  let note;
  if (item.ours === null) {
    f.getRange(`H${row}`).values = [['n/r']];
    f.getRange(`B${row}`).values = [[index - 0.23]];
    note = index === 2 ? 'No independent full-pipeline WER; prior value was online' : 'No reproduction WER for the voice arm';
  } else {
    note = `Matched dates; ${item.ours_days}d/${item.ours_trials} trials`;
    if (index === 5) note += '; 50 words; OPT tuned on eval';
    else if (index === 6 || index === 7) note += '; 125k; OPT tuned on eval';
    else if (index === 8 || index === 9) note += '; online dates; test=validation';
    else if (index === 0) note += '; bigram';
    else note += '; 5-gram + OPT';
  }
  f.getRange(`I${row}`).values = [[note]];
  f.getRange(`I${row}`).format.horizontalAlignment = 'left';
}
workbook.recalculate();
console.log((await workbook.inspect({ kind: 'region', sheetId: 'f_repro_fullWER', range: 'C2:I8', maxChars: 2300 })).ndjson);
const after = await workbook.render({ sheetName: 'f_repro_fullWER', range: 'C1:I23', scale: 1 });
await fs.writeFile(path.join(stage, 'source_after.png'), new Uint8Array(await after.arrayBuffer()));
await (await SpreadsheetFile.exportXlsx(workbook)).save(path.join(stage, 'SourceData_Figure2.xlsx'));
console.log('Updated only the reproduction values and their scope notes in Source Data.');
