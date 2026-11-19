import {cp, mkdir, readFile, readdir, rm, writeFile} from 'node:fs/promises';

const root = new URL('../', import.meta.url);
const site = new URL('.site/', root);
await rm(site, {recursive: true, force: true});
await cp(new URL('demo_output/', root), site, {recursive: true});
const backtest = new URL('backtest/', site);
await cp(new URL('backtest/web/', root), backtest, {recursive: true});
const modules = new URL('python/backtest/', backtest);
await mkdir(modules, {recursive: true});
await cp(new URL('modelo.py', root), new URL('python/modelo.py', backtest));
for (const file of await readdir(new URL('backtest/', root))) {
  if (file.endsWith('.py')) await cp(new URL('backtest/' + file, root), new URL(file, modules));
}
const panel = new URL('PAINEL_INTERATIVO.html', site);
const content = await readFile(panel, 'utf8');
if (!content.includes('<body>')) throw new Error('Painel sem elemento body.');
const nav = '<nav aria-label="Navegação principal" style="display:flex;gap:24px;padding:16px;'
  + 'font:16px system-ui,sans-serif;border-bottom:1px solid #d9e1eb">'
  + '<a href="/" aria-current="page">Painel</a><a href="/backtest/">Backtest</a></nav>';
await writeFile(panel, content.replace('<body>', '<body>' + nav));

const plotly = [...content.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)]
  .find(([, script]) => script.includes('* plotly.js v'));
if (!plotly) throw new Error('Biblioteca Plotly não encontrada no painel.');
await writeFile(new URL('plotly.js', backtest), plotly[1]);
