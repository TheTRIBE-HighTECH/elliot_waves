"""Build signals using confirmed pivots and trailing windows."""
import numpy as np

from modelo import FIB_BANDS, best_fibonacci_impulses, confirmed_pivots, scale_exponent


def trailing_hurst(prices, axis, window=256, step=16):
    spectrum = []
    medium = np.full(len(prices), np.nan)
    for end in range(window, len(prices) + 1, step):
        for band, lags in FIB_BANDS.items():
            h, r2 = scale_exponent(prices[end-window:end], axis[end-window:end], lags)
            spectrum.append({'index': end - 1, 'band': band, 'h': h, 'r2': r2})
            if band.startswith('média'):
                medium[end-1:end+step-1] = h
    return medium, spectrum


def pivot_events(prices, reversal):
    pivots = confirmed_pivots(prices, reversal)
    events, entries, history = [], {}, []
    for pivot in pivots:
        history.append(pivot)
        known = pivot.confirmed_at
        event = {'extreme': pivot.index, 'confirmed': known, 'tradable': known + 1,
                 'kind': pivot.kind, 'state': 'CONFIRMADA'}
        if len(history) >= 3:
            a, b, c = history[-3:]
            rise = prices[b.index] - prices[a.index]
            if a.kind == 'trough' and rise > 0:
                ratio = (prices[b.index] - prices[c.index]) / rise
                valid = 0 < ratio < 1
                event['wave2'] = {'indices': [a.index, b.index, c.index], 'ratio': ratio,
                                  'state': 'CANDIDATA' if valid else 'INVALIDADA'}
                if valid:
                    if .382 <= ratio <= .786:
                        event['wave2']['state'] = 'CONFIRMADA'
                    entries[known] = (c.index, a.index, b.index, ratio)
        if len(history) >= 6:
            impulses = best_fibonacci_impulses(prices, history[-6:])
            if impulses:
                impulse = impulses[0]
                event['impulse'] = impulse
        # A/B/C after a confirmed bullish impulse. Other corrective families are not assumed.
        for completed_legs in (1, 2, 3):
            if len(events) < completed_legs or 'impulse' not in events[-completed_legs]:
                continue
            correction = history[-completed_legs-1:]
            levels = [prices[pivot.index] for pivot in correction]
            valid = completed_legs < 2 or levels[2] < levels[0]
            if completed_legs == 3:
                valid = valid and levels[3] < levels[1]
            state = ('INVALIDADA' if not valid else 'CONFIRMADA' if completed_legs == 3
                     else 'CANDIDATA')
            event.setdefault('abc', []).append({'indices': [p.index for p in correction],
                                                 'state': state, 'legs': completed_legs})
        events.append(event)
    return events, entries


def signals(model, candles, axis, h, waves, config):
    close = candles.close.to_numpy()
    speed = candles.clock_speed.to_numpy()
    result = {}
    if model in ('C', 'D'):
        for i, (correction, start, top, ratio) in waves.items():
            if axis[correction] - axis[start] < config['wave_duration_min']:
                continue
            if config['use_fibonacci'] and not .382 <= ratio <= .786:
                continue
            stop = candles.low.iloc[correction] * (1 - config['stop_buffer'])
            multiple = config['target_multiple'] if config['use_fibonacci'] else 1.0
            target = close[correction] + multiple * (close[top] - close[start])
            if stop < close[i] < target:
                result[i] = (float(stop), float(target), 'onda_3')
    else:
        lookback = config['lookback']
        starts = np.searchsorted(axis, axis - lookback, side='right') - 1
        momentum = np.full(len(close), np.nan)
        indices = np.flatnonzero(starts >= 0)
        momentum[indices] = close[indices] / close[starts[indices]] - 1
        for i in np.flatnonzero((momentum[1:] > 0) & (momentum[:-1] <= 0)) + 1:
            result[int(i)] = (close[i] * (1 - config['trend_stop']),
                              close[i] * (1 + config['target_multiple'] * config['trend_stop']),
                              'virada_momento')
    return {i: order for i, order in result.items()
            if (config['hurst_min'] is None or h[i] >= config['hurst_min'])
            and (model != 'D' or config['speed_min'] is None or speed[i] >= config['speed_min'])}
