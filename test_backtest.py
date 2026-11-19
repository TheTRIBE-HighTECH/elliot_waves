import unittest

import numpy as np
import pandas as pd

from backtest.data import archive_paths, prepare
from backtest.engine import execute
from backtest.experiment import DEFAULTS, run, splits
from backtest.strategy import pivot_events, trailing_hurst
from modelo import information_clock


def bars(n):
    index = np.arange(n)
    close = 100 + np.sin(index / 4) * 3 + index / 100
    return pd.DataFrame({'timestamp': pd.date_range('2025-01-01', periods=n, freq='h', tz='UTC'),
                         'open': close, 'high': close * 1.003, 'low': close * .997, 'close': close,
                         'volume': 100.0, 'quote_volume': 10_000 + index % 92,
                         'num_trades': index % 31, 'taker_buy_volume': 55.0})


class InitiationTests(unittest.TestCase):
    def test_confirmation_and_untradeable_pivot_until_reversal(self):
        prices = np.log([100, 105, 110, 108, 104, 105, 108])
        events, signals = pivot_events(prices, .025)
        self.assertEqual([(e['extreme'], e['confirmed'], e['tradable']) for e in events],
                         [(0, 1, 2), (2, 4, 5), (4, 6, 7)])
        self.assertEqual(set(signals), {6})
        for i in range(2, len(prices)):
            prefix, _ = pivot_events(prices[:i], .025)
            self.assertEqual(prefix, [e for e in events if e['confirmed'] < i])

    def test_information_clock_initial_and_prefix_invariance(self):
        frame = bars(500)
        clock = information_clock(frame)
        np.testing.assert_allclose(clock.tau_minutes[:60], np.arange(60) * 60)
        self.assertTrue((np.diff(clock.tau_minutes) > 0).all())
        np.testing.assert_allclose(np.diff(clock.tau_minutes), clock.clock_speed.iloc[1:] * 60)
        prefix = information_clock(frame.iloc[:320])
        np.testing.assert_allclose(prefix.tau_minutes, clock.tau_minutes.iloc[:320])
        prices = np.log(frame.close)
        full, _ = trailing_hurst(prices.to_numpy(), clock.tau_minutes.to_numpy() / 60)
        partial, _ = trailing_hurst(prices.iloc[:320].to_numpy(), prefix.tau_minutes.to_numpy() / 60)
        np.testing.assert_allclose(full[:320], partial, equal_nan=True)

    def test_mixed_timestamps_utc_and_data_validation(self):
        frame = bars(24)
        t = frame.timestamp
        frame['timestamp'] = [str(int(value.timestamp() * multiplier)) for value, multiplier
                              in zip(t, [1, 1_000, 1_000_000] * 8)]
        opts = ('2025-01-01', '2025-01-01', '1h', [])
        ordered, meta = prepare(frame.iloc[::-1], *opts)
        self.assertTrue(meta['reordered'])
        pd.testing.assert_series_equal(ordered.timestamp, t)
        with self.assertRaisesRegex(ValueError, 'duplicados'):
            prepare(pd.concat([frame, frame.iloc[:1]]), *opts)
        with self.assertRaisesRegex(ValueError, 'ausentes'):
            prepare(frame.iloc[:-1], *opts)
        paths = list(archive_paths('BTCUSDT', '1h', '2025-01-31', '2025-02-28'))
        self.assertTrue(paths[0].startswith('daily/'))
        self.assertTrue(paths[1].startswith('monthly/'))


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'open': [99., 100., 100.], 'high': [150., 115., 102.],
                                   'low': [50., 85., 97.], 'close': [100., 100., 100.]})
        self.config = {**DEFAULTS, 'capital': 1000, 'risk': .1, 'fee_bps': 0,
                       'spread_bps': 0, 'slippage_bps': 0}

    def test_next_open_stop_first_and_target_policy(self):
        signals = {0: (90, 110, 'test')}
        stop = execute(self.frame, signals, 0, 3, self.config)
        self.assertEqual(stop['equity'], [1000, 900, 900])
        self.assertEqual(stop['orders'][0]['execution_index'], 1)
        self.assertEqual(stop['trades'][0]['net'], -100)
        target = execute(self.frame, signals, 0, 3, {**self.config, 'ambiguity': 'target'})
        self.assertEqual(target['trades'][0]['net'], 100)

    def test_costs_reconcile_gross_and_net(self):
        report = execute(self.frame, {0: (90, 110, 'test')}, 0, 3,
                         {**self.config, 'fee_bps': 10, 'spread_bps': 4, 'slippage_bps': 2})
        trade, = report['trades']
        self.assertAlmostEqual(trade['entry'], 100.04)
        self.assertAlmostEqual(trade['exit'], 89.964)
        self.assertAlmostEqual(trade['gross'] - trade['net'], report['costs'])
        self.assertAlmostEqual(report['equity'][-1], 1000 + trade['net'])

    def test_passive_entry_does_not_assume_high_after_fill(self):
        frame = self.frame.copy()
        frame.loc[1, ['open', 'low']] = [104, 98]
        report = execute(frame, {0: (90, 110, 'test')}, 0, 3,
                         {**self.config, 'maker_bps': 5, 'order_type': 'limit'})
        trade, = report['trades']
        self.assertEqual(trade['entry'], 100)
        self.assertEqual(trade['exit_index'], 2)
        self.assertAlmostEqual(trade['entry_fee'], trade['quantity'] * .05)

    def test_embargo_and_tuning_do_not_use_future_data(self):
        config = {**DEFAULTS, 'interval': '1h'}
        frame = bars(600)
        meta = {'sha256': 'synthetic-tests'}
        baseline = run(frame, meta, config)
        changed = frame.copy()
        start = baseline['segments']['test'][0]
        changed.loc[start:, ['close', 'open', 'high', 'low']] *= .7
        alternate = run(changed, meta, config)
        self.assertEqual(baseline, run(frame, meta, config))
        self.assertEqual(baseline['selected_settings'], alternate['selected_settings'])
        self.assertEqual(baseline['validation'], alternate['validation'])
        self.assertEqual(set(baseline['selected_settings']), set('ABCD'))
        for model in 'ABCD':
            self.assertEqual(len(baseline['results'][model]['equity']), len(frame) - start)
            old = [s for s in baseline['signals'][model] if s['signal_index'] < start]
            new = [s for s in alternate['signals'][model] if s['signal_index'] < start]
            self.assertEqual(old, new)
        for fold in baseline['walk_forward']:
            self.assertEqual(fold['out'][0] - fold['train'][1], config['embargo'])
            self.assertLessEqual(fold['out'][1], start - config['embargo'])
        with self.assertRaisesRegex(ValueError, 'Histórico curto'):
            splits(20, config)


if __name__ == '__main__':
    unittest.main()
