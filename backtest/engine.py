"""Spot long-only event replay. Candle timestamps represent opening times."""
import numpy as np


def execute(candles, signals, start, stop, config, hold=False):
    cash = initial = config['capital']
    gross_cash = initial
    rate = config['fee_bps'] / 10_000
    maker_rate = config['maker_bps'] / 10_000
    impact = (config['spread_bps'] / 2 + config['slippage_bps']) / 10_000
    position = None
    trades, orders, equity, gross_equity = [], [], [], []
    balances, position_value = [], []
    exposed = turnover = gross_turnover = costs = 0.0

    def exit_trade(i, raw, reason):
        nonlocal cash, gross_cash, costs, turnover, gross_turnover, position
        quantity = position['quantity']
        fill = raw * (1 - impact)
        fee = fill * quantity * rate
        cash += fill * quantity - fee
        gross_cash += raw * quantity
        costs += (raw - fill) * quantity + fee
        turnover += fill * quantity
        gross_turnover += raw * quantity
        gross = (raw - position['raw_entry']) * quantity
        net = (fill - position['entry']) * quantity - fee - position['entry_fee']
        trades.append({**position, 'exit_index': i, 'raw_exit': raw, 'exit': fill,
                       'exit_fee': fee, 'reason': reason, 'net': net, 'gross': gross})
        position = None

    ohlc = candles[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
    for i in range(start, stop):
        opening, high, low, close = ohlc[i]
        pending = signals.get(i - 1) if i > start else None
        if hold and i == start:
            pending = (0, float('inf'), 'buy_hold')
        if pending:
            stop_price, target, reason = pending
            passive = not hold and config['order_type'] == 'limit'
            limit = ohlc[i - 1, 3] if passive else None
            raw = limit if passive else opening
            fill = raw if passive else opening * (1 + impact)
            entry_rate = maker_rate if passive else rate
            order = {'signal_index': i - 1 if not hold else None, 'execution_index': i, 'reason': reason}
            order.update({'type': 'limit' if passive else 'market', 'limit': limit})
            if position:
                order['status'] = 'posição_aberta'
            elif passive and not low < limit < opening:
                order['status'] = 'limite_não_executado'
            elif not stop_price < raw <= fill < target:
                order['status'] = 'preço_fora_dos_limites'
            else:
                max_qty = cash / (fill * (1 + entry_rate))
                worst_stop = stop_price * (1 - impact)
                unit_risk = fill * (1 + entry_rate) - worst_stop * (1 - rate)
                quantity = max_qty if hold else min(max_qty, cash * config['risk'] / unit_risk)
                fee = fill * quantity * entry_rate
                cash = max(0, cash - fill * quantity - fee)
                gross_cash -= raw * quantity
                turnover += fill * quantity
                gross_turnover += raw * quantity
                costs += fee + (fill - raw) * quantity
                position = {'signal_index': order['signal_index'], 'entry_index': i, 'raw_entry': raw, 'entry_type': order['type'],
                            'entry': fill, 'entry_fee': fee, 'quantity': quantity,
                            'stop': None if hold else stop_price, 'target': None if hold else target}
                order['status'] = 'executada'
            orders.append(order)
        if position:
            exposed += 1
            if not hold:
                stop_price, target = position['stop'], position['target']
                just_filled_limit = position['entry_type'] == 'limit' and position['entry_index'] == i
                if just_filled_limit and low <= stop_price:
                    exit_trade(i, stop_price, 'stop')
                elif just_filled_limit:
                    pass  # The candle high may have occurred before the passive fill.
                elif opening <= stop_price:
                    exit_trade(i, opening, 'gap_stop')
                elif opening >= target:
                    exit_trade(i, opening, 'gap_alvo')
                elif low <= stop_price and (high < target or config['ambiguity'] == 'stop'):
                    exit_trade(i, stop_price, 'stop')
                elif high >= target:
                    exit_trade(i, target, 'alvo')
            if position and i == stop - 1:
                exit_trade(i, close, 'fim_do_segmento')
        quantity = position['quantity'] if position else 0
        balances.append(cash)
        position_value.append(quantity * close)
        equity.append(cash + quantity * close)
        gross_equity.append(gross_cash + quantity * close)
    if stop - 1 in signals and not hold:
        orders.append({'signal_index': stop - 1, 'execution_index': None,
                       'reason': signals[stop - 1][2], 'status': 'fim_do_segmento'})
    return {'equity': equity, 'gross_equity': gross_equity, 'trades': trades,
            'orders': orders, 'cash': balances, 'position_value': position_value,
            'costs': costs, 'turnover': turnover, 'gross_turnover': gross_turnover, 'exposed': exposed}
