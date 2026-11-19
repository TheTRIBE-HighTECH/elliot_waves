"""Validate UTC candles and retain the provenance of the exact sample."""
import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from modelo import ARCHIVE_COLS, REQUIRED

SYMBOLS = ('BTCUSDT', 'ETHUSDT', 'SOLUSDT')
INTERVALS = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600, '4h': 14400, '1d': 86400}
FIELDS = ['timestamp', 'open', 'high', 'low', *REQUIRED[1:]]
MAX_BARS = 100_000
ARCHIVE = 'https://data.binance.vision/data/spot/'


def sample_bounds(start, end, interval):
    if interval not in INTERVALS:
        raise ValueError('Timeframe inválido.')
    first, last = [pd.Timestamp(value, tz='UTC') for value in (start, end)]
    if pd.isna(first) or pd.isna(last) or any(value != value.normalize() for value in (first, last)):
        raise ValueError('Use datas UTC sem horário.')
    stop = last + pd.Timedelta(days=1)
    count = (stop - first).total_seconds() / INTERVALS[interval]
    if not 0 < count <= MAX_BARS:
        raise ValueError(f'Selecione um período válido de até {MAX_BARS:,} candles.')
    return first, stop


def archive_paths(symbol, interval, start, end):
    if symbol not in SYMBOLS:
        raise ValueError('Ativo inválido.')
    first, stop = sample_bounds(start, end, interval)
    day = first
    while day < stop:
        month_end = day + pd.offsets.MonthBegin()
        if day.day == 1 and month_end <= stop:
            kind, period, day = 'monthly', day.strftime('%Y-%m'), month_end
        else:
            kind, period, day = 'daily', day.strftime('%Y-%m-%d'), day + pd.Timedelta(days=1)
        yield f'{kind}/klines/{symbol}/{interval}/{symbol}-{interval}-{period}.zip'


def read_archive(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        files = [file for file in archive.infolist() if file.filename.lower().endswith('.csv')]
        if len(files) != 1 or files[0].file_size > 30_000_000:
            raise ValueError('Arquivo Binance inválido.')
        with archive.open(files[0]) as csv:
            frame = pd.read_csv(csv, header=None, dtype={0: str})
    if frame.shape[1] != len(ARCHIVE_COLS):
        raise ValueError('Formato Binance inválido.')
    frame.columns = ARCHIVE_COLS
    if frame.timestamp.iloc[0].lower() in {'open time', 'open_time', 'timestamp'}:
        frame = frame.iloc[1:]
    return frame


def download(path):
    with urllib.request.urlopen(ARCHIVE + path, timeout=35) as response:
        return response.read()


def read_csv(content):
    return pd.read_csv(io.BytesIO(content), dtype={'timestamp': str, 'open_time': str})


def prepare(frame, start, end, interval, sources):
    first, stop = sample_bounds(start, end, interval)
    frame = frame.rename(columns={'open_time': 'timestamp'}) if 'timestamp' not in frame else frame
    missing = sorted(set(FIELDS) - set(frame.columns))
    if missing:
        raise ValueError('Colunas ausentes: ' + ', '.join(missing))
    frame = frame[FIELDS].copy()
    raw = frame.timestamp.astype('string').str.strip()
    numeric = raw.str.fullmatch(r'[0-9]+', na=False)
    dates = pd.to_datetime(raw.where(~numeric), format='mixed', utc=True, errors='coerce')
    for digits, unit in [(10, 's'), (13, 'ms'), (16, 'us')]:
        mask = numeric & (raw.str.len() == digits)
        dates.loc[mask] = pd.to_datetime(raw.loc[mask].astype('int64'), unit=unit, utc=True)
    if dates.isna().any():
        raise ValueError('Timestamp inválido. Use ISO 8601 ou epoch em s, ms ou µs.')
    frame['timestamp'] = dates
    frame = frame.loc[dates.ge(first) & dates.lt(stop)].copy()
    if frame.empty:
        raise ValueError('Nenhum candle no período selecionado.')
    reordered = not frame.timestamp.is_monotonic_increasing
    if frame.timestamp.duplicated().any():
        raise ValueError('Há timestamps duplicados no período.')
    frame = frame.sort_values('timestamp').reset_index(drop=True)
    expected = pd.date_range(first, stop, freq=pd.Timedelta(seconds=INTERVALS[interval]), inclusive='left')
    absent = expected.difference(frame.timestamp)
    off_grid = pd.DatetimeIndex(frame.timestamp).difference(expected)
    if len(absent) or len(off_grid):
        raise ValueError(f'Histórico irregular: {len(absent)} candles ausentes e {len(off_grid)} fora da grade UTC.')
    frame[FIELDS[1:]] = frame[FIELDS[1:]].apply(pd.to_numeric, errors='coerce')
    if (not np.isfinite(frame[FIELDS[1:]].to_numpy(dtype=float)).all()
            or (frame[['open', 'high', 'low', 'close']] <= 0).any().any()
            or (frame[REQUIRED[2:]] < 0).any().any()
            or (frame.taker_buy_volume > frame.volume + 1e-9).any()
            or (frame.high < frame[['open', 'close', 'low']].max(axis=1)).any()
            or (frame.low > frame[['open', 'close']].min(axis=1)).any()):
        raise ValueError('Valores OHLCV ou fluxo de ordens inválidos.')
    digest = hashlib.sha256(frame.to_csv(index=False, float_format='%.17g').encode()).hexdigest()
    return frame, {'sources': sources, 'sha256': digest, 'schema': 1, 'rows': len(frame),
                   'start_utc': first.isoformat(), 'end_exclusive_utc': stop.isoformat(),
                   'interval': interval, 'reordered': reordered}


def load(config, csv_path=None):
    if csv_path:
        payload = Path(csv_path).read_bytes()
        frame = read_csv(payload)
        sources = [{'type': 'csv', 'name': Path(csv_path).name,
                    'sha256': hashlib.sha256(payload).hexdigest()}]
    else:
        frames, sources = [], []
        for path in archive_paths(config['symbol'], config['interval'], config['start'], config['end']):
            payload = download(path)
            frames.append(read_archive(payload))
            sources.append({'type': 'binance_spot', 'url': ARCHIVE + path,
                            'sha256': hashlib.sha256(payload).hexdigest()})
        frame = pd.concat(frames, ignore_index=True)
    return prepare(frame, config['start'], config['end'], config['interval'], sources)
