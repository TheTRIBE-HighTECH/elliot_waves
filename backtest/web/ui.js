const byId = (id) => document.getElementById(id);
const form = byId('experiment');
const status = byId('status');
let worker, result, serialized;
const format = new Intl.NumberFormat('pt-BR', {maximumFractionDigits: 4});
const display = (value) => value == null ? '—' : typeof value === 'number' ? format.format(value) : value;

function stop(message) {
  worker?.terminate();
  worker = undefined;
  byId('run').disabled = false;
  byId('cancel').hidden = true;
  status.textContent = message;
}

byId('cancel').onclick = () => stop('Execução cancelada.');

form.onsubmit = async (event) => {
  event.preventDefault();
  if (worker) return;
  const options = {};
  for (const [key, value] of new FormData(form)) {
    const kind = form.elements.namedItem(key).type;
    options[key] = kind === 'number' ? (value === '' ? null : Number(value)) : value;
  }
  options.score_weights = options.score_weights.split(',').map((number) => Number(number.trim()));
  options.risk /= 100;
  options.reversal /= 100;
  const file = byId('csv').files[0];
  if (file?.size > 30_000_000) {
    status.textContent = 'O CSV deve ter até 30 MB.';
    return;
  }
  const job = new Worker('/backtest/worker.js');
  worker = job;
  byId('run').disabled = true;
  byId('cancel').hidden = false;
  status.textContent = 'Iniciando experimento';
  job.onerror = (error) => stop(error.message || 'Falha no processamento.');
  job.onmessage = ({data}) => {
    const message = typeof data === 'string' ? JSON.parse(data) : data;
    if (message.progress) status.textContent = message.progress;
    if (message.error) stop(message.error);
    if (message.output) {
      serialized = message.output;
      result = JSON.parse(serialized);
      stop('Experimento concluído.');
      try { render(); } catch (error) { status.textContent = error.message; }
    }
  };
  try {
    const csv = file ? await file.arrayBuffer() : undefined;
    if (worker === job) job.postMessage({options, csv, csvName: file?.name}, csv ? [csv] : []);
  } catch (error) {
    if (worker === job) stop(error.message);
  }
};

function table(target, headers, rows) {
  const head = document.createElement('thead');
  const header = head.insertRow();
  for (const key of headers) {
    const cell = document.createElement('th');
    cell.scope = 'col';
    cell.textContent = key;
    header.append(cell);
  }
  const body = document.createElement('tbody');
  for (const values of rows) {
    const row = body.insertRow();
    for (const value of values) row.insertCell().textContent = display(value);
  }
  target.replaceChildren(head, body);
}

const metricLabels = {
  cumulative_return: 'Retorno acumulado', annualized_return: 'Retorno anualizado', sharpe: 'Sharpe',
  sortino: 'Sortino', max_drawdown: 'Drawdown máximo', calmar: 'Calmar', profit_factor: 'Profit factor',
  win_rate: 'Taxa de acerto', expectancy: 'Expectativa por operação', trade_count: 'Operações',
  average_duration_bars: 'Duração média (candles)', exposure: 'Exposição', turnover: 'Turnover',
  execution_cost: 'Custo de execução', final_equity: 'Patrimônio final',
};
const percentMetrics = new Set(['cumulative_return', 'annualized_return', 'max_drawdown', 'win_rate', 'exposure']);

function showMetrics() {
  const results = Object.entries(result.results);
  const mode = byId('mode').value;
  table(byId('metrics'), ['Métrica', ...results.map(([name]) => name)],
    Object.entries(metricLabels).map(([key, label]) => [label, ...results.map(([, item]) => {
      const value = item[mode][key];
      return value != null && percentMetrics.has(key) ? format.format(100 * value) + '%' : value;
    })]));
}

function showScience() {
  const {science} = result;
  const target = byId('science');
  const interval = (value) => value?.map(display).join(' a ') || 'amostra insuficiente';
  const elliott = science.elliott_comparison;
  const summaries = [
    `C × D · ${elliott.shared_signals} sinais em comum, ${elliott.only_c} só em C e ${elliott.only_d} só em D.`
      + (elliott.identical_trades && elliott.identical_equity
        ? ' As operações e a curva de patrimônio foram idênticas nesta amostra.' : ''),
    'H1 · Dispersão relativa das durações: ' + science.H1.status + '. IC 95%: ' + interval(science.H1.interval95) + '.',
    'H2 · Geometria: ' + science.H2.status + '. ' + science.H2.reason,
    'H3 · Diferença de retorno diário líquido de D frente aos controles.',
  ].map((text) => Object.assign(document.createElement('p'), {textContent: text}));
  const comparison = document.createElement('table');
  table(comparison, ['Controle', 'Diferença média (bps/dia)', 'IC 95% (bps/dia)', 'Resultado'],
    Object.entries(science.H3).map(([name, row]) => [name, row.mean_daily_difference * 10000,
      interval(row.interval95?.map((value) => value * 10000)), row.status]));
  target.replaceChildren(...summaries, comparison);
  byId('geometry').textContent = JSON.stringify(result.geometry_score, null, 2);
  byId('details').textContent = JSON.stringify({
    segmentos: result.segments, validação: result.validation, walkForward: result.walk_forward,
    custos: science.cost_sensitivity, parâmetros: science.parameter_sensitivity,
    ablações: science.ablations, regimes: science.regimes,
    sinaisPreditivos: science.predictive_labels, incertezaHurst: science.hurst_uncertainty,
  }, null, 2);
}

function plot(parent, title, traces, layout = {}) {
  const container = document.createElement('div');
  parent.append(container);
  Plotly.newPlot(container, traces, {title: {text: title}, margin: {t: 55, l: 68, r: 30, b: 50},
    autosize: true, ...layout}, {responsive: true, displaylogo: false});
}

function draw() {
  const charts = result.charts;
  const [first, last] = result.segments.test;
  const [model, run] = [byId('chart-model').value, result.results[byId('chart-model').value]];
  const ohlc = charts.ohlc.slice(first, last);
  const time = charts.time.slice(first, last);
  const tau = charts.tau.slice(first, last);
  const plots = byId('plots');
  for (const child of plots.children) window.Plotly?.purge(child);
  plots.replaceChildren();
  table(byId('trades'), ['Entrada UTC', 'Saída (candle UTC)', 'Preço entrada', 'Preço saída', 'Qtd.', 'P&L líquido', 'Motivo'],
    run.trades.map((trade) => [trade.entry_utc, trade.exit_candle_utc, trade.entry, trade.exit,
      trade.quantity, trade.net, trade.reason]));
  if (!window.Plotly) {
    plots.textContent = 'A biblioteca Plotly não carregou. Métricas, operações e exportação continuam disponíveis.';
    return;
  }
  const column = (index) => ohlc.map((row) => row[index]);
  const tradeMarker = (axis, index, price, name, color) => ({type: 'scatter', mode: 'markers',
    x: run.trades.map((trade) => axis[trade[index] - first]), y: run.trades.map((trade) => trade[price]),
    name, marker: {color, size: 8}});
  const pivots = result.pivots.filter((item) => item.extreme >= first && item.confirmed < last);
  const waveX = [], waveY = [];
  for (const wave of charts.stability) {
    waveX.push(...wave.indices.map((i) => charts.time[i]), null);
    waveY.push(...wave.indices.map((i) => charts.ohlc[i][3]), null);
  }
  plot(plots, 'Preço cronológico', [{x: time, open: column(0), high: column(1), low: column(2),
    close: column(3), type: 'candlestick', name: 'OHLC'},
    {x: pivots.map((item) => charts.time[item.extreme]), y: pivots.map((item) => charts.ohlc[item.extreme][3]),
      text: pivots.map((item) => 'Confirmação: ' + item.confirmed_utc), type: 'scatter', mode: 'markers', name: 'Pivôs'},
    {x: waveX, y: waveY, mode: 'lines', name: 'Impulsos Elliott'},
    tradeMarker(time, 'entry_index', 'entry', 'Entradas', '#059669'),
    tradeMarker(time, 'exit_index', 'exit', 'Saídas', '#dc2626')],
    {xaxis: {rangeslider: {visible: false}}});
  plot(plots, 'Mesmo preço em tempo informacional', [{x: tau, y: column(3), name: 'Fechamento'},
    tradeMarker(tau, 'entry_index', 'entry', 'Entradas', '#059669'),
    tradeMarker(tau, 'exit_index', 'exit', 'Saídas', '#dc2626')], {xaxis: {title: {text: 'τ (minutos)'}}});
  plot(plots, 'Velocidade informacional', [{x: time, y: charts.speed.slice(first, last), name: 'dτ/dt'}],
    {shapes: [{type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 1, y1: 1, line: {dash: 'dot'}}]});
  for (const [clock, series] of Object.entries(charts.hurst)) {
    const filtered = series.filter((row) => row.index >= first && row.index < last);
    const bands = [...new Set(filtered.map((row) => row.band))];
    const groups = bands.map((band) => filtered.filter((row) => row.band === band));
    if (groups.length) plot(plots, 'Hurst · ' + (clock === 'calendar' ? 'cronológico' : 'informacional'),
      [{x: groups[0].map((row) => charts.time[row.index]), y: bands,
        z: groups.map((rows) => rows.map((row) => row.h)), type: 'heatmap', colorscale: 'Viridis'}]);
  }
  const comparisons = Object.entries(result.results).slice(0, 4);
  plot(plots, 'Patrimônio líquido', comparisons.map(([name, row]) => ({x: time, y: row.equity, name})));
  plot(plots, 'Drawdown', comparisons.map(([name, row]) => ({x: time, y: row.drawdown, name})),
    {yaxis: {tickformat: '.1%'}});
  plot(plots, 'Distribuição de P&L por operação · ' + model, ['gross', 'net'].map((key) =>
    ({x: run.trades.map((trade) => trade[key]), type: 'histogram', opacity: .6, name: key === 'gross' ? 'Bruto' : 'Líquido'})),
  {barmode: 'overlay'});
  const normalizedDurations = (key) => charts.stability.flatMap((item) => {
    const sum = item[key].reduce((a, b) => a + b, 0);
    return item[key].map((value) => value / sum);
  });
  plot(plots, 'Proporções de preço dos impulsos confirmados', [{
    x: charts.stability.flatMap(() => ['Onda 2/1', 'Onda 3/1', 'Onda 4/3', 'Onda 5/1']),
    y: charts.stability.flatMap((item) => item.ratios),
    type: 'box', name: 'Pivôs compartilhados nos dois relógios',
  }]);
  plot(plots, 'Duração relativa dos impulsos', ['calendar', 'information'].map((key) =>
    ({y: normalizedDurations(key), type: 'box', name: key === 'calendar' ? 'Cronológico' : 'Informacional'})));
}

function render() {
  byId('results').hidden = false;
  byId('sample').textContent = `${result.config.symbol} · ${result.data.rows} candles · ${result.data.start_utc} até ${result.data.end_exclusive_utc} (exclusivo) · execução ${result.run_id.slice(0, 12)}`;
  byId('chart-model').replaceChildren(...Object.keys(result.results).map((name) => new Option(name, name)));
  byId('chart-model').value = 'D';
  showMetrics();
  showScience();
  draw();
}

byId('mode').onchange = showMetrics;
byId('chart-model').onchange = draw;
byId('export').onclick = () => {
  const url = URL.createObjectURL(new Blob([serialized], {type: 'application/json'}));
  const link = document.createElement('a');
  link.href = url;
  link.download = `backtest-${result.run_id.slice(0, 12)}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
