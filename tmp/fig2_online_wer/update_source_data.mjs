import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

const stage = path.dirname(fileURLToPath(import.meta.url));
const input = 'D:/repositories/20261012_Comment_on_Recent_Progress_of_Language_BCI/latex/source_data/SourceData_Figure2.xlsx';
const summary = JSON.parse(await fs.readFile(path.join(stage, 'online_summary.json'), 'utf8'));
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(input));
const before = await workbook.render({ sheetName: 'c_online_WER', range: 'C1:I12', scale: 1 });
await fs.writeFile(path.join(stage, 'source_before.png'), new Uint8Array(await before.arrayBuffer()));
console.log((await workbook.inspect({ kind: 'region', sheetId: 'c_online_WER', range: 'C2:I4', maxChars: 1800 })).ndjson);

// Preserve empty source cells as blanks through the import/export cycle.
const original = JSON.parse(await fs.readFile(path.join(stage, 'original_workbook_values.json'), 'utf8'));
for (const [name, rows] of Object.entries(original)) {
  const sheet = workbook.worksheets.getItem(name);
  for (const [r, row] of rows.entries()) {
    for (const [c, value] of row.entries()) {
      if (value === null) sheet.getCell(r, c).values = [[null]];
    }
  }
}

const c = workbook.worksheets.getItem('c_online_WER');
const f = workbook.worksheets.getItem('f_repro_fullWER');
for (const [index, item] of summary.entries()) {
  const row = index + 2;
  const comparatorRow = 3 + 2 * index;
  const numbers = [[item.mean, item.sd, item.sd]];
  c.getRange(`E${row}:G${row}`).values = numbers;
  c.getRange(`E${row}:G${row}`).setNumberFormat('0.00');
  c.getRange(`E${row}:G${row}`).format.horizontalAlignment = 'right';
  const scope = item.note.match(/(\d+) of (\d+) days/);
  const days = scope ? `${scope[1]}/${scope[2]} days` : null;
  const note = days
    ? item.sd === null ? `Single-day word-count WER; ${days}` : `Daily counts; equal-day mean +/- sample SD; ${days}`
    : 'Listener transcription median (90 sampled trials)';
  c.getRange(`I${row}`).values = [[note]];
  c.getRange(`I${row}`).format.horizontalAlignment = 'left';
  f.getRange(`E${comparatorRow}:G${comparatorRow}`).values = numbers;
  f.getRange(`E${comparatorRow}:G${comparatorRow}`).setNumberFormat('0.00');
  f.getRange(`E${comparatorRow}:G${comparatorRow}`).format.horizontalAlignment = 'right';
  const comparisonNote = days
    ? `Online WER from (c); ${item.sd === null ? 'single day' : 'equal-day mean +/- SD'}; ${days}`
    : 'Online WER from (c); listener transcription median';
  f.getRange(`I${comparatorRow}`).values = [[comparisonNote]];
  f.getRange(`I${comparatorRow}`).format.horizontalAlignment = 'left';
}
workbook.recalculate();
console.log((await workbook.inspect({ kind: 'region', sheetId: 'c_online_WER', range: 'C2:I4', maxChars: 1800 })).ndjson);
const after = await workbook.render({ sheetName: 'c_online_WER', range: 'C1:I12', scale: 1 });
await fs.writeFile(path.join(stage, 'source_after.png'), new Uint8Array(await after.arrayBuffer()));
const paired = await workbook.render({ sheetName: 'f_repro_fullWER', range: 'C1:I23', scale: 1 });
await fs.writeFile(path.join(stage, 'source_f_after.png'), new Uint8Array(await paired.arrayBuffer()));
await (await SpreadsheetFile.exportXlsx(workbook)).save(path.join(stage, 'SourceData_Figure2.xlsx'));
console.log('Updated the online WER values and scope notes in panels c and f.');
