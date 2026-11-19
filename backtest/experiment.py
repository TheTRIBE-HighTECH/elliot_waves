"""Four-model out-of-sample experiment with reproducible chronological folds."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from modelo import information_clock
from .data import INTERVALS, load
from .engine import execute
from .statistics import block_ci, conclusion, geometry_scores, metrics
from .strategy import pivot_events, signals, trailing_hurst

VERSION = '1.1.0'
DEFAULTS = {'capital': 10_000, 'risk': .01, 'fee_bps': 10, 'maker_bps': 10, 'spread_bps': 2,
            'slippage_bps': 2, 'ambiguity': 'stop', 'reversal': .006, 'target_multiple': 1.618,
            'stop_buffer': .001, 'trend_stop': .02, 'lookback': 20, 'hurst_min': None,
            'speed_min': .8, 'embargo': 3, 'walk_folds': 3, 'seed': 73,
            'use_fibonacci': True, 'holdout': .2, 'order_type': 'market',
            'wave_duration_min': 2, 'score_weights': [1, 1, 1, 1, 1]}
MODELS = ('A', 'B', 'C', 'D', 'Buy and Hold', 'Médias móveis', 'D sem filtro de velocidade')


def configure(options):
    config = {**DEFAULTS, **options}
    limits = {'capital': (1, 1e12), 'risk': (.0001, .2), 'reversal': (.0001, .5),
              'target_multiple': (.1, 10), 'stop_buffer': (0, .2), 'trend_stop': (.001, .5),
              'lookback': (4, 500), 'wave_duration_min': (0, 25), 'embargo': (0, 100), 'walk_folds': (1, 6), 'seed': (0, 2**32 - 1),
              'fee_bps': (0, 500), 'maker_bps': (0, 500), 'spread_bps': (0, 500),
              'slippage_bps': (0, 500), 'holdout': (.15, .3)}
    for key, (low, high) in limits.items():
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not low <= value <= high:
            raise ValueError(f'Parâmetro inválido: {key}.')
        if key in {'lookback', 'embargo', 'walk_folds', 'seed'} and int(value) != value:
            raise ValueError(f'Use um inteiro para {key}.')
        if key in {'lookback', 'embargo', 'walk_folds', 'seed'}:
            config[key] = int(value)
    for key, low, high in [('hurst_min', 0, 1), ('speed_min', .35, 3.5)]:
        value = config[key]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int))
                                  or not low <= value <= high):
            raise ValueError(f'Parâmetro inválido: {key}.')
    if config['ambiguity'] not in ('stop', 'target'):
        raise ValueError('Política intrabar inválida.')
    if config['order_type'] not in ('market', 'limit'):
        raise ValueError('Tipo de ordem inválido.')
    weights = config['score_weights']
    if (not isinstance(weights, list) or len(weights) != 5
            or any(not isinstance(w, (int, float)) or not 0 <= w <= 10 for w in weights)
            or sum(weights) == 0):
        raise ValueError('Defina cinco pesos geométricos entre 0 e 10.')
    return config


def splits(n, config):
    embargo = config['embargo']
    validation = int(n * .55)
    test = int(n * (1 - config['holdout']))
    folds = []
    edges = np.linspace(validation, test - embargo, config['walk_folds'] + 1).astype(int)
    for start, stop in zip(edges[:-1], edges[1:]):
        folds.append({'train': [0, int(start) - embargo], 'out': [int(start), int(stop)]})
    segments = {'train': [0, validation - embargo], 'validation': [validation, test - embargo],
                'test': [test, n], 'walk_forward': folds}
    if min(segments['train'][1], *(b - a for a, b in (segments['validation'], segments['test'])),
           *(fold['out'][1] - fold['out'][0] for fold in folds)) < 12:
        raise ValueError('Histórico curto para treino, validação, teste e embargo.')
    return segments


def moving_average(candles, config):
    close = candles.close
    fast = close.rolling(max(2, config['lookback'] // 4)).mean()
    slow = close.rolling(config['lookback']).mean()
    above = fast > slow
    return {int(i): (price * (1 - config['trend_stop']),
                     price * (1 + config['target_multiple'] * config['trend_stop']), 'cruzamento')
            for i, price in close[above & ~above.shift(fill_value=False) & slow.shift().notna()].items()}


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    return None if isinstance(value, float) and not math.isfinite(value) else value


def run(frame, provenance, options, progress=lambda *_: None):
    stages = []
    def report(message):
        stages.append(message)
        progress(message)

    config = configure(options)
    split = splits(len(frame), config)
    report('Calculando relógios e Hurst')
    candles = information_clock(frame)
    seconds = INTERVALS[config['interval']]
    bars_per_year = 365.25 * 86400 / seconds
    logprice = np.log(candles.close.to_numpy())
    timestamps = candles.timestamp.dt.strftime('%Y-%m-%dT%H:%M:%SZ').tolist()
    axes = {key: candles[column].to_numpy() / (seconds / 60)
            for key, column in [('calendar', 'calendar_minutes'), ('information', 'tau_minutes')]}
    h, spectra = {}, {}
    for clock, axis in axes.items():
        h[clock], spectra[clock] = trailing_hurst(logprice, axis)
    # Every A–D model receives the same three candidate settings and execution budget.
    def variants(settings):
        pivot_key = settings['reversal']
        if pivot_key not in pivots:
            pivots[pivot_key] = pivot_events(logprice, pivot_key)
        events, waves = pivots[pivot_key]
        return {model: signals(model, candles, axes[clock], h[clock], waves, settings)
                for model, clock in [('A', 'calendar'), ('B', 'information'),
                                     ('C', 'calendar'), ('D', 'information')]}, events

    pivots = {}
    grid = [{**config, 'lookback': max(4, round(config['lookback'] * factor)),
             'reversal': min(.5, config['reversal'] * factor)} for factor in (.75, 1, 1.25)]
    candidates = [variants(settings)[0] for settings in grid]

    def replay(name, entries, start, stop, settings):
        result = execute(candles, entries, start, stop, settings, hold=name == 'Buy and Hold')
        net, dd, returns = metrics(result['equity'], result['trades'], settings['capital'],
                                   bars_per_year, 'net', result)
        gross, _, _ = metrics(result['gross_equity'], result['trades'], settings['capital'],
                              bars_per_year, 'gross', result)
        return {**result, 'net': net, 'gross': gross, 'drawdown': dd}, returns

    def select(model, train):
        # Prefer the base parameters when training drawdown-adjusted returns tie.
        return max((1, 0, 2), key=lambda idx:
                   training_score(execute(candles, candidates[idx][model], *train, grid[idx]), config))

    report('Ajustando somente nos segmentos de treino')
    selected = {model: select(model, split['train']) for model in MODELS[:4]}
    selected_settings = {model: grid[idx] for model, idx in selected.items()}
    entries = {model: candidates[idx][model] for model, idx in selected.items()}
    settings_d = selected_settings['D']
    entries.update({'Buy and Hold': {}, 'Médias móveis': moving_average(candles, config),
                    'D sem filtro de velocidade': variants({**settings_d, 'speed_min': None})[0]['D']})
    validation, results, daily, walks = {}, {}, {}, []
    start, stop = split['test']
    for model in MODELS:
        report(f'Avaliando {model} fora da amostra')
        settings = selected_settings.get(model, settings_d if model == 'D sem filtro de velocidade' else config)
        validation[model] = replay(model, entries[model], *split['validation'], settings)[0]['net']
        results[model], returns = replay(model, entries[model], start, stop, settings)
        daily[model] = day_returns(returns, candles.timestamp.iloc[start:stop])
    for fold in split['walk_forward']:
        row = {}
        for model in MODELS[:4]:
            index = select(model, fold['train'])
            row[model] = {'grid_index': index,
                          'net': replay(model, candidates[index][model], *fold['out'], grid[index])[0]['net']}
        walks.append({**fold, 'models': row})

    _, events = variants(settings_d)
    for event in events:
        event['extreme_utc'] = timestamps[event['extreme']]
        interval = np.timedelta64(seconds, 's')
        event['confirmed_utc'] = (candles.timestamp.iloc[event['confirmed']] + interval).isoformat()
        event['tradable_utc'] = event['confirmed_utc']
    train_end = split['train'][1]
    stability = []
    for event in events:
        if 'impulse' not in event or event['confirmed'] < start:
            continue
        indices = event['impulse']['pivot_indices']
        if indices[0] < start:
            continue
        stability.append({'indices': indices, 'ratios': event['impulse']['ratios'],
                          'calendar': np.diff(axes['calendar'][indices]).tolist(),
                          'information': np.diff(axes['information'][indices]).tolist()})

    h1_values, last_end = [], -1
    for item in stability:
        if item['indices'][0] < last_end:
            continue
        last_end = item['indices'][-1]
        h1_values.append(dispersion(item['calendar']) - dispersion(item['information']))
    h1_ci = block_ci(h1_values, 3, config['seed'])
    comparisons = {}
    for model in ('A', 'B', 'C', 'Buy and Hold', 'Médias móveis', 'D sem filtro de velocidade'):
        diff = daily['D'] - daily[model]
        ci = block_ci(diff, 5, config['seed'])
        comparisons[model] = {'mean_daily_difference': float(np.mean(diff)), 'interval95': ci,
                              'status': conclusion(ci)}

    predictive = {}
    horizon = 5
    price = candles.close.to_numpy()
    opening = candles.open.to_numpy()
    for model in MODELS[:4]:
        # Label only complete, non-overlapping opportunities in the final test.
        indices, next_index = [], start
        for index in sorted(entries[model]):
            entry = index + 1
            if index >= start and entry >= next_index and entry + horizon <= stop:
                indices.append(entry)
                next_index = entry + horizon
        points = np.asarray(indices, dtype=int)
        observed = price[points + horizon - 1] / opening[points] - 1
        predictive[model] = {'horizon_bars': horizon, 'observations': len(points),
                             'hit_rate': float(np.mean(observed > 0)) if len(points) else None,
                             'mean_return': float(np.mean(observed)) if len(points) else None,
                             'signals': [{'signal_index': point - 1, 'forward_return': value}
                                         for point, value in zip(points, observed)]}
    sensitivity = {}
    for factor in (.5, 1.5, 2):
        stressed = {**settings_d, **{key: settings_d[key] * factor
                    for key in ('fee_bps', 'maker_bps', 'spread_bps', 'slippage_bps')}}
        sensitivity[str(factor)] = replay('D', entries['D'], start, stop, stressed)[0]['net']
    ablations = {}
    for label, override in [('sem_hurst', {'hurst_min': None}),
                            ('sem_fibonacci', {'use_fibonacci': False}),
                            ('sem_relogio', {})]:
        ablated = {**settings_d, **override}
        model = 'C' if label == 'sem_relogio' else 'D'
        ablations[label] = replay(model, variants(ablated)[0][model], start, stop, ablated)[0]['net']

    train_vol = np.std(np.diff(logprice[:train_end]))
    vol = candles.close.pct_change().rolling(20).std().to_numpy()
    test_returns = daily['D']
    high_regime = vol[start:stop] > train_vol
    d_equity = np.r_[config['capital'], results['D']['equity']]
    bar_returns = np.diff(d_equity) / d_equity[:-1]
    regimes = {name: {'bars': int(mask.sum()), 'mean_bar_return': float(bar_returns[mask].mean()) if mask.any() else None}
               for name, mask in [('alta_volatilidade', high_regime),
                                  ('baixa_volatilidade', (~high_regime) & np.isfinite(vol[start:stop]))]}
    audit = {model: [{'signal_index': i, 'confirmed_close_utc':
                      (candles.timestamp.iloc[i] + np.timedelta64(seconds, 's')).isoformat(),
                      'tradable_index': i + 1, 'stop': order[0], 'target': order[1], 'reason': order[2],
                      'breakout': order[3] if len(order) > 3 else None}
                     for i, order in entries[model].items()] for model in MODELS}
    for replay_result in results.values():
        for trade in replay_result['trades']:
            trade['entry_utc'] = timestamps[trade['entry_index']]
            trade['exit_candle_utc'] = timestamps[trade['exit_index']]
    report('Consolidando validação científica e auditoria')
    hurst_uncertainty = {}
    for clock, spectrum in spectra.items():
        bands = {}
        for band in dict.fromkeys(row['band'] for row in spectrum):
            values = [row['h'] for row in spectrum if row['band'] == band and row['index'] >= start
                      and np.isfinite(row['h'])]
            # Consecutive H windows share 240 of 256 bars. Preserve at least 16 sampled windows.
            bands[band] = {'observations': len(values), 'interval95_mean': block_ci(values, 16, config['seed'])}
        hurst_uncertainty[clock] = bands
    run_id = hashlib.sha256(json.dumps([VERSION, provenance['sha256'], config], sort_keys=True).encode()).hexdigest()
    return clean({'run_id': run_id, 'version': VERSION, 'config': config, 'data': provenance,
                  'stages': stages, 'segments': split, 'selected_settings': selected_settings, 'validation': validation,
                  'walk_forward': walks, 'results': results, 'signals': audit, 'pivots': events,
                  'charts': {'time': timestamps, 'tau': candles.tau_minutes.tolist(),
                             'speed': candles.clock_speed.tolist(),
                             'ohlc': candles[['open', 'high', 'low', 'close']].to_numpy().tolist(),
                             'hurst': spectra, 'stability': stability},
                  'geometry_score': geometry_scores(events, axes, h, train_end, start, config['score_weights']),
                  'science': {'hurst_uncertainty': hurst_uncertainty, 'H1': {'sample': len(h1_values), 'interval95': h1_ci, 'status': conclusion(h1_ci)},
                              'H2': {'status': 'inconclusivo', 'reason': 'Os mesmos pivôs preservam razões de preço sob reparametrização.'},
                              'H3': comparisons, 'predictive_labels': predictive, 'days': len(test_returns), 'cost_sensitivity': sensitivity,
                              'ablations': ablations, 'regimes': regimes,
                              'parameter_sensitivity': [
                                  replay('D', candidate['D'], start, stop, grid[idx])[0]['net']
                                  for idx, candidate in enumerate(candidates)]}})


def training_score(result, config):
    equity = np.r_[config['capital'], result['equity']]
    drawdown = (1 - equity / np.maximum.accumulate(equity)).max()
    return equity[-1] / equity[0] - 1 - drawdown


def day_returns(returns, timestamps):
    days = timestamps.dt.floor('D').to_numpy()
    _, inverse = np.unique(days, return_inverse=True)
    totals = np.ones(int(inverse.max()) + 1)
    np.multiply.at(totals, inverse, 1 + returns)
    return totals - 1


def dispersion(values):
    return np.std(values) / np.mean(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--interval', default='1h')
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--options', type=Path, help='JSON with strategy and execution overrides')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    options = json.loads(args.options.read_text()) if args.options else {}
    config = {**options, 'symbol': args.symbol, 'interval': args.interval,
              'start': args.start, 'end': args.end}
    frame, provenance = load(config, args.csv)
    result = run(frame, provenance, config, print)
    args.out.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f'{result["run_id"]} → {args.out}')


if __name__ == '__main__':
    main()
