const runtimeUrl = 'https://cdn.jsdelivr.net/pyodide/v0.29.3/full/';
let runtime;
const pythonFiles = ['modelo.py', 'backtest/__init__.py', 'backtest/data.py',
  'backtest/strategy.py', 'backtest/engine.py', 'backtest/statistics.py', 'backtest/experiment.py'];

async function initialize() {
  importScripts(runtimeUrl + 'pyodide.js');
  const py = await loadPyodide({indexURL: runtimeUrl});
  await py.loadPackage(['numpy', 'pandas', 'matplotlib']);
  py.FS.mkdirTree('/home/pyodide/backtest');
  await Promise.all(pythonFiles.map(async (file) => {
    const response = await fetch('/backtest/python/' + file);
    if (!response.ok) throw new Error('Falha ao carregar ' + file);
    py.FS.writeFile('/home/pyodide/' + file, await response.text());
  }));
  py.FS.chdir('/home/pyodide');
  return py;
}

self.onmessage = async ({data}) => {
  try {
    self.postMessage({progress: 'Carregando o motor Python'});
    runtime ||= initialize();
    const py = await runtime;
    py.globals.set('options_json', JSON.stringify(data.options));
    py.globals.set('csv_name', data.csvName || '');
    if (data.csv) py.FS.writeFile('/tmp/sample.csv', new Uint8Array(data.csv));
    const output = await py.runPythonAsync(`
import hashlib
import json
from pathlib import Path
import pandas as pd
from js import postMessage
from pyodide.http import pyfetch
from backtest.data import ARCHIVE, archive_paths, prepare, read_archive, read_csv
from backtest.experiment import configure, run

def progress(message):
    postMessage(json.dumps({'progress': message}))

config = configure(json.loads(options_json))
if csv_name:
    payload = Path('/tmp/sample.csv').read_bytes()
    frame = read_csv(payload)
    sources = [{'type': 'csv', 'name': csv_name, 'sha256': hashlib.sha256(payload).hexdigest()}]
else:
    frames, sources = [], []
    paths = list(archive_paths(config['symbol'], config['interval'], config['start'], config['end']))
    for idx, path in enumerate(paths, 1):
        progress(f'Baixando arquivo Binance {idx}/{len(paths)}')
        response = await pyfetch('/backtest/binance/' + path)
        if not response.ok:
            raise ValueError(f'Binance HTTP {response.status}: {path}. Verifique o período ou importe um CSV.')
        payload = await response.bytes()
        frames.append(read_archive(payload))
        sources.append({'type': 'binance_spot', 'url': ARCHIVE + path, 'sha256': hashlib.sha256(payload).hexdigest()})
    frame = pd.concat(frames, ignore_index=True)
frame, provenance = prepare(frame, config['start'], config['end'], config['interval'], sources)
json.dumps(run(frame, provenance, config, progress), ensure_ascii=False, allow_nan=False)
`);
    self.postMessage({output});
  } catch (error) {
    self.postMessage({error: error.message});
  }
};
