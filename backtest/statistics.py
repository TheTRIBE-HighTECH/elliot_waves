"""Metrics and seeded paired block bootstrap."""
import numpy as np


def metrics(equity, trades, capital, bars_per_year, field, replay):
    equity = np.asarray(equity)
    returns = np.diff(np.r_[capital, equity]) / np.r_[capital, equity[:-1]]
    drawdown = equity / np.maximum.accumulate(np.r_[capital, equity])[1:] - 1
    change = float(equity[-1] / capital - 1)
    years = len(equity) / bars_per_year
    annual = float(np.expm1(np.log1p(change) / years)) if years >= 0.25 and change > -1 else None
    downside = np.sqrt(np.mean(np.minimum(0, returns) ** 2))
    std = np.std(returns, ddof=1) if len(returns) > 1 else 0
    pnl = np.array([trade[field] for trade in trades])
    losses = -pnl[pnl < 0].sum()
    return {'cumulative_return': change, 'annualized_return': annual,
            'sharpe': returns.mean() / std * np.sqrt(bars_per_year) if std > 0 else None,
            'sortino': returns.mean() / downside * np.sqrt(bars_per_year) if downside > 0 else None,
            'max_drawdown': float(drawdown.min()),
            'calmar': annual / -drawdown.min() if annual is not None and drawdown.min() < 0 else None,
            'profit_factor': pnl[pnl > 0].sum() / losses if losses > 0 else None,
            'win_rate': float((pnl > 0).mean()) if len(pnl) else None,
            'expectancy': float(pnl.mean()) if len(pnl) else None,
            'trade_count': len(trades),
            'average_duration_bars': float(np.mean([t['exit_index'] - t['entry_index'] + 1 for t in trades])) if trades else None,
            'exposure': replay['exposed'] / len(equity), 'turnover': replay['turnover' if field == 'net' else 'gross_turnover'] / capital,
            'execution_cost': replay['costs'] if field == 'net' else 0,
            'final_equity': float(equity[-1])}, drawdown.tolist(), returns


def block_ci(values, block, seed, statistic=np.mean):
    values = np.asarray(values)
    if len(values) < 4 * block:
        return None
    rng = np.random.default_rng(seed)
    width = int(np.ceil(len(values) / block))
    offsets = np.arange(block)
    estimates = []
    for _ in range(300):
        starts = rng.integers(0, len(values) - block + 1, size=width)
        indices = (starts[:, None] + offsets).ravel()[:len(values)]
        estimates.append(statistic(values[indices]))
    return np.quantile(estimates, [.025, .975]).tolist()


def conclusion(ci, minimum=0):
    if ci is None or ci[0] <= minimum <= ci[1]:
        return 'inconclusivo'
    return 'positivo' if ci[0] > minimum else 'negativo'


def geometry_scores(events, axes, hurst, train_end, test_start, weights):
    """Training empirical percentiles, reported with raw components."""
    labels = ('fibonacci_error', 'structural_validity', 'duration_dispersion',
              'hurst_persistence', 'confirmation_lag')
    result = {}
    impulses = [event for event in events if 'impulse' in event]
    for clock, axis in axes.items():
        rows = []
        for event in impulses:
            index = event['confirmed']
            pivots = event['impulse']['pivot_indices']
            durations = np.diff(axis[pivots])
            value = hurst[clock][index]
            raw = [1 - event['impulse']['score'], 1.0, np.std(durations) / np.mean(durations),
                   max(0.0, float(value) - .5) if np.isfinite(value) else None,
                   axis[index] - axis[event['extreme']]]
            rows.append((index, pivots[0], raw))
        training = [[row[2][i] for row in rows if row[0] < train_end and row[2][i] is not None]
                    for i in range(len(labels))]
        test = []
        for index, first, raw in rows:
            if first < test_start:
                continue
            normalized = []
            for i, (value, history) in enumerate(zip(raw, training)):
                if value is None or not history:
                    normalized.append(None)
                    continue
                values = np.asarray(history)
                better = values < value if i in (1, 3) else values > value
                normalized.append(float(np.mean(better + .5 * (values == value))))
            valid = [(value, weight) for value, weight in zip(normalized, weights)
                     if value is not None and weight > 0]
            composition = (sum(value * weight for value, weight in valid) / sum(weight for _, weight in valid)
                           if valid else None)
            test.append({'confirmed_index': index, 'raw': dict(zip(labels, raw)),
                         'normalized': dict(zip(labels, normalized)), 'score': composition})
        result[clock] = {'training_counts': dict(zip(labels, map(len, training))), 'test': test}
    return result
