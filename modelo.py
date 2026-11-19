#!/usr/bin/env python3
"""Elliott–Fibonacci / information-time research prototype.

Three observable outputs:
(1) market price versus calendar time;
(2) identical market price versus endogenous order-flow information time;
(3) multi-scale Hurst exponent estimates on both clocks using Fibonacci lags.

Research-only prototype. Elliott candidates are detected AFTER confirmation,
not by identifying a turning point before it is observed.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

FIB_BANDS = {
    'curta (1–13)': np.array([1, 2, 3, 5, 8, 13]),
    'média (3–34)': np.array([3, 5, 8, 13, 21, 34]),
    'longa (8–89)': np.array([8, 13, 21, 34, 55, 89]),
}
ARCHIVE_COLS = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'num_trades', 'taker_buy_volume',
    'taker_buy_quote_volume', 'ignore'
]
REQUIRED = ['timestamp', 'close', 'volume', 'quote_volume', 'num_trades', 'taker_buy_volume']


def epoch_to_datetime(values: pd.Series) -> pd.Series:
    nums = pd.to_numeric(values, errors='coerce')
    typical = nums.dropna().abs().median()
    if np.isfinite(typical) and typical > 1e14:
        unit = 'us'  # Binance spot archives from 2025 onwards
    elif np.isfinite(typical) and typical > 1e11:
        unit = 'ms'
    elif np.isfinite(typical) and typical > 1e8:
        unit = 's'
    else:
        raise ValueError('Unexpected epoch timestamp format')
    return pd.to_datetime(nums, unit=unit, utc=True)


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'timestamp' not in df.columns and 'open_time' in df.columns:
        df = df.rename(columns={'open_time': 'timestamp'})
    absent = [name for name in REQUIRED if name not in df.columns]
    if absent:
        raise ValueError('Missing CSV columns: ' + ', '.join(absent))
    if pd.api.types.is_numeric_dtype(df['timestamp']):
        df['timestamp'] = epoch_to_datetime(df['timestamp'])
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    for name in REQUIRED[1:]:
        df[name] = pd.to_numeric(df[name], errors='coerce')
    df = (df[REQUIRED].dropna().drop_duplicates('timestamp', keep='last')
          .sort_values('timestamp').reset_index(drop=True))
    df = df[(df.close > 0) & (df.volume >= 0) & (df.quote_volume >= 0) &
            (df.num_trades >= 0) & (df.taker_buy_volume >= 0)].reset_index(drop=True)
    if len(df) < 750:
        raise ValueError('At least 750 observations are needed (long-scale Hurst window).')
    dt = df.timestamp.diff().dt.total_seconds().dropna().to_numpy() / 60
    if np.any(dt <= 0):
        raise ValueError('Timestamps must be strictly increasing.')
    if (np.max(dt) / np.min(dt)) > 3:
        print('Warning: irregular observation intervals. Check missing bars.', file=sys.stderr)
    return df


def download_binance(symbol: str, interval: str, period: str, periodicity: str = 'monthly') -> pd.DataFrame:
    if not symbol.isalnum() or not interval.isalnum():
        raise ValueError('Symbol and interval must be alphanumeric.')
    if periodicity not in {'daily', 'monthly'}:
        raise ValueError('periodicity must be daily or monthly')
    filename = f'{symbol}-{interval}-{period}.zip'
    url = f'https://data.binance.vision/data/spot/{periodicity}/klines/{symbol}/{interval}/{filename}'
    print('Downloading original Binance data:', url, flush=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'InformationClockResearch/0.1'})
    with urllib.request.urlopen(req, timeout=35) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith('.csv')]
        if len(csv_names) != 1:
            raise ValueError('Expected precisely one CSV in Binance archive.')
        content = archive.read(csv_names[0])
    df = pd.read_csv(io.BytesIO(content), header=None)
    # Some external derivatives of the public archive contain headers.
    if str(df.iat[0, 0]).lower() in {'open time', 'open_time', 'timestamp'}:
        df = df.iloc[1:].reset_index(drop=True)
    if df.shape[1] < 12:
        raise ValueError('Binances kline archive must contain 12 columns.')
    df = df.iloc[:, :12]
    df.columns = ARCHIVE_COLS
    return standardize(df)


def robust_past_z(series: pd.Series, window: int = 300, min_periods: int = 60) -> np.ndarray:
    """Lagged, robust rolling z-score: statistics never contain present/future observations."""
    value = pd.Series(np.asarray(series, dtype=float))
    median = value.rolling(window, min_periods=min_periods).median().shift(1)
    deviation = (value - median).abs()
    mad = deviation.rolling(window, min_periods=min_periods).median().shift(1)
    z = (value - median) / (1.4826 * mad + 0.15)
    return z.fillna(0).clip(-3, 3).to_numpy(dtype=float)


def information_clock(df: pd.DataFrame) -> pd.DataFrame:
    """Causal activity-based clock, not an actual measure of Shannon news information."""
    df = df.copy()
    base = df['volume'].to_numpy(dtype=float)
    buyer = df['taker_buy_volume'].to_numpy(dtype=float)
    imbalance = np.zeros(len(df))
    np.divide(np.abs(2 * buyer - base), base, out=imbalance, where=base > 0)
    imbalance = np.clip(imbalance, 0, 1)
    z_trades = robust_past_z(np.log1p(df['num_trades']))
    z_volume = robust_past_z(np.log1p(df['quote_volume']))
    z_imbalance = robust_past_z(pd.Series(imbalance))
    score = 0.45 * z_trades + 0.35 * z_volume + 0.20 * z_imbalance
    v = np.clip(np.exp(np.clip(0.50 * score, -1.2, 1.2)), 0.35, 3.5)
    dt = df.timestamp.diff().dt.total_seconds().to_numpy(dtype=float) / 60
    if len(dt) > 1:
        dt[0] = np.median(dt[1:])
    else:
        dt[0] = 1.
    # tau starts at zero. The first bar's clock rate is observable only at its close.
    tau = np.zeros(len(df))
    tau[1:] = np.cumsum(v[1:] * dt[1:])
    df['order_imbalance'] = imbalance
    df['clock_speed'] = v
    df['tau_minutes'] = tau
    df['calendar_minutes'] = np.r_[0, np.cumsum(dt[1:])]
    return df


def synthetic_data(n: int = 5000, seed: int = 73) -> pd.DataFrame:
    """Exogenous activity bursts + Elliott-like motifs in operational time + noise."""
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    arrival = (0.25 * np.sin(2 * np.pi * i / 780) +
               1.4 * np.exp(-0.5 * ((i - n * 0.26) / 120)**2) +
               1.2 * np.exp(-0.5 * ((i - n * 0.68) / 145)**2))
    count = rng.poisson(130 * np.exp(arrival)).astype(int)
    volume = rng.lognormal(mean=2.2 + 0.45 * arrival, sigma=0.4, size=n)
    ratio = np.clip(0.50 + 0.19 * np.sin(2*np.pi*i/310) +
                    rng.normal(0, 0.09, size=n), 0.01, 0.99)
    df = pd.DataFrame({
        'timestamp': pd.date_range('2026-10-01', periods=n, freq='min', tz='UTC'),
        'close': np.ones(n) * 66000.0,  # replaced below
        'volume': volume,
        'quote_volume': volume * 66000,
        'num_trades': count,
        'taker_buy_volume': volume * ratio
    })
    df = information_clock(df)
    tau = df.tau_minutes.to_numpy()
    # An eight-segment 1-2-3-4-5-A-B-C wave template, repeated in tau-time.
    knot_t = np.array([0, .13, .22, .40, .52, .66, .76, .83, 1])
    knot_x = np.array([0, 1, .382, 2.0, 1.382, 2.382, 1.50, 1.85, .9])
    phase = (tau % 720) / 720
    cycle = np.floor(tau / 720)
    motif = np.interp(phase, knot_t, knot_x) * 0.0060
    drift = 0.0018 * cycle + 0.000002 * tau
    # Brownian increments have exogenous, clock-dependent activity.
    increments = rng.normal(0, 0.00018 * np.sqrt(df.clock_speed.to_numpy()), n)
    noise = np.cumsum(increments)
    x = np.log(66000.0) + motif + drift + noise
    df['close'] = np.exp(x)
    df['quote_volume'] = df.volume * df.close
    return standardize(df[REQUIRED])


def scale_exponent(x: np.ndarray, axis: np.ndarray, lags: np.ndarray) -> tuple[float, float]:
    """Estimate variance-time H and R² using only observations in the window."""
    variances, valid_lags = [], []
    for lag in lags:
        usable = axis >= axis[0] + lag
        if usable.sum() < 80:
            continue
        past = np.interp(axis[usable] - lag, axis, x)
        delta = x[usable] - past
        var = np.var(delta, ddof=1)
        if np.isfinite(var) and var > 1e-15:
            variances.append(var)
            valid_lags.append(lag)
    if len(valid_lags) < 4:
        return np.nan, np.nan
    lx = np.log(valid_lags)
    ly = np.log(variances)
    slope, intercept = np.polyfit(lx, ly, 1)
    predicted = slope * lx + intercept
    denom = np.sum((ly - ly.mean())**2)
    r2 = 1 - np.sum((ly - predicted)**2)/denom if denom > 0 else np.nan
    return float(slope / 2), float(r2)


def hurst_spectra(df: pd.DataFrame, window: int = 700, step: int = 60) -> pd.DataFrame:
    x = np.log(df.close.to_numpy(dtype=float))
    cal = df.calendar_minutes.to_numpy(dtype=float)
    inf = df.tau_minutes.to_numpy(dtype=float)
    if len(df) < window + 40:
        window = min(640, len(df) - 30)
    output = []
    for end in range(window, len(df)+1, step):
        start = end - window
        for label, lags in FIB_BANDS.items():
            hc, rc = scale_exponent(x[start:end], cal[start:end]-cal[start], lags)
            hi, ri = scale_exponent(x[start:end], inf[start:end]-inf[start], lags)
            output.append({
                'end_idx': end-1, 'timestamp': str(df.timestamp.iloc[end-1]),
                'scale': label, 'H_calendar': hc, 'H_information': hi,
                'R2_calendar': rc, 'R2_information': ri
            })
    return pd.DataFrame(output)


@dataclass
class Pivot:
    index: int
    confirmed_at: int
    kind: str


def confirmed_pivots(logprice: np.ndarray, reversal: float = 0.004) -> list[Pivot]:
    """Online ZigZag: each extremum is only known once a reverse move confirms it."""
    direction = 0
    hi_idx = lo_idx = 0
    extreme_idx = 0
    result: list[Pivot] = []
    for i in range(1, len(logprice)):
        if direction == 0:
            if logprice[i] > logprice[hi_idx]:
                hi_idx = i
            if logprice[i] < logprice[lo_idx]:
                lo_idx = i
            if logprice[i] - logprice[lo_idx] >= reversal:
                result.append(Pivot(lo_idx, i, 'trough'))
                direction = 1
                extreme_idx = i
            elif logprice[hi_idx] - logprice[i] >= reversal:
                result.append(Pivot(hi_idx, i, 'peak'))
                direction = -1
                extreme_idx = i
        elif direction == 1:
            if logprice[i] >= logprice[extreme_idx]:
                extreme_idx = i
            elif logprice[extreme_idx] - logprice[i] >= reversal:
                result.append(Pivot(extreme_idx, i, 'peak'))
                extreme_idx = i
                direction = -1
        else:
            if logprice[i] <= logprice[extreme_idx]:
                extreme_idx = i
            elif logprice[i] - logprice[extreme_idx] >= reversal:
                result.append(Pivot(extreme_idx, i, 'trough'))
                extreme_idx = i
                direction = 1
    return result


def best_fibonacci_impulses(logprice: np.ndarray, pivots: list[Pivot], top_k: int = 5) -> list[dict]:
    """Candidates satisfy impulse rules; similarity score is descriptive, not predictive."""
    fib_targets = [(.382,.5,.618,.786), (1.,1.618,2.618),(.236,.382,.5),(.618,1.,1.618)]
    found = []
    for a in range(len(pivots)-5):
        p = pivots[a:a+6]
        if [q.kind for q in p] != ['trough','peak','trough','peak','trough','peak']:
            continue
        y = np.array([logprice[q.index] for q in p])
        d = np.diff(y)
        if not np.all(d * np.array([1,-1,1,-1,1]) > 0):
            continue
        if not (y[2] > y[0] and y[3] > y[1] and y[4] > y[1] and
                y[5] > y[3] and d[2] >= min(d[0],d[4])):
            continue
        ratios = np.array([abs(d[1]/d[0]), abs(d[2]/d[0]),
                           abs(d[3]/d[2]), abs(d[4]/d[0])])
        distances = [min(abs(np.log(r/t)) for t in ts) for r,ts in zip(ratios,fib_targets)]
        score = float(np.exp(-np.mean(np.square(distances))/0.09))
        found.append({'pivot_indices':[q.index for q in p],
                      'known_at':p[-1].confirmed_at, 'score':score,
                      'ratios':ratios.tolist()})
    found.sort(key=lambda v:v['score'], reverse=True)
    return found[:top_k]


def set_plot_style():
    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'grid.alpha': .19,
    })


def plot_outputs(df: pd.DataFrame, spec: pd.DataFrame, candidates: list[dict], dest: Path, source_label: str) -> None:
    set_plot_style()
    dates = df.timestamp
    prices = df.close
    info = df.tau_minutes
    figs = []

    fig, ax = plt.subplots(figsize=(12,4), layout='constrained')
    ax.plot(dates, prices, color='#2563eb', lw=1.2)
    ax.set(title=f'1 • Preço × tempo cronológico — {source_label}', ylabel='Preço', xlabel='Data (UTC)')
    ax.grid(True)
    for c in candidates[:2]:
        pi = c['pivot_indices']
        ax.scatter(dates.iloc[pi], prices.iloc[pi], s=15, color='#d97706', zorder=4)
        ax.plot(dates.iloc[pi], prices.iloc[pi], color='#d97706', lw=1, alpha=.7)
    fig.savefig(dest/'01_preco_cronologico.png', dpi=180)
    figs.append(fig)

    fig, ax = plt.subplots(figsize=(12,4), layout='constrained')
    ax.plot(info, prices, color='#0f766e', lw=1.2)
    ax.set(title=f'2 • Mesmo preço × relógio de fluxo de negociações — {source_label}',
           ylabel='Preço', xlabel='Minutos informacionais (τ)')
    ax.grid(True)
    fig.savefig(dest/'02_preco_informacional.png', dpi=180)
    figs.append(fig)

    bands = list(FIB_BANDS.keys())
    stamps = spec.drop_duplicates('end_idx').end_idx.to_numpy(dtype=int)
    values = []
    for clock in ('H_calendar','H_information'):
        matrix = (spec.pivot(index='scale', columns='end_idx', values=clock)
                  .reindex(bands).reindex(columns=stamps).to_numpy(dtype=float))
        values.append(matrix)
    fig, axs = plt.subplots(2, 1, figsize=(12,5.8), sharex=True, layout='constrained')
    for ax, matrix, label in zip(axs, values, ('Tempo cronológico','Tempo informacional')):
        im = ax.imshow(matrix, aspect='auto', vmin=0.1, vmax=0.9, cmap='viridis', interpolation='nearest')
        ax.set_yticks(range(len(bands)), bands)
        ax.set_ylabel(label)
        ax.set_title('3 • Espectro local de Hurst — bandas de lags Fibonacci' if ax is axs[0] else '')
        ax.grid(False)
    xloc = np.unique(np.round(np.linspace(0,len(stamps)-1,6)).astype(int))
    axs[-1].set_xticks(xloc, [str(dates.iloc[stamps[i]])[:16] for i in xloc], rotation=15)
    axs[-1].set_xlabel('Fim da janela retrospectiva (UTC)')
    fig.colorbar(im, ax=list(axs), label='Estimativa H (cores saturadas fora de 0,1–0,9)', shrink=.83)
    fig.savefig(dest/'03_espectro_hurst.png', dpi=180)
    figs.append(fig)

    fig = plt.figure(figsize=(13,11), layout='constrained')
    gs = fig.add_gridspec(4, 2, height_ratios=[1,1,.65,.8])
    ax1 = fig.add_subplot(gs[0,:]); ax2 = fig.add_subplot(gs[1,:])
    ax3 = fig.add_subplot(gs[2,:]); ax4 = fig.add_subplot(gs[3,0]); ax5=fig.add_subplot(gs[3,1])
    ax1.plot(dates, prices, lw=1., color='#2563eb')
    ax1.set(title='01 | Preço vs. tempo cronológico', ylabel='Preço (USDT)'); ax1.grid(True)
    ax2.plot(info,prices,lw=1,color='#0f766e')
    ax2.set(title='02 | Mesmo preço vs. tempo informacional',ylabel='Preço (USDT)',xlabel='τ (minutos operacionais)');ax2.grid(True)
    ax3.plot(dates,df.clock_speed, color='#ca8a04',lw=.75)
    ax3.axhline(1,linestyle='--',color='gray',lw=1)
    ax3.set(title='Velocidade do relógio calculada apenas com fluxo de negociações',ylabel='dτ/dt');ax3.grid(True)
    norm=Normalize(0.1,.9)
    for ax, matrix,label in zip((ax4,ax5),values,('Cronológico','Informacional')):
        ax.imshow(matrix,aspect='auto',norm=norm,cmap='viridis',interpolation='nearest')
        ax.set_yticks(range(len(bands)),['Curta','Média','Longa']);ax.set_xticks([])
        ax.set(title='03 | Hurst por escala — '+label)
    fig.suptitle('Elliott–Fibonacci × relógio informacional | '+source_label,fontsize=14,weight='bold')
    fig.savefig(dest/'PAINEL_COMPLETO.png',dpi=160)
    figs.append(fig)
    for fig in figs:
        plt.close(fig)



def write_interactive_dashboard(df: pd.DataFrame, spec: pd.DataFrame, candidates: list[dict], dest: Path, source: str) -> None:
    """Self-contained HTML with interactive axes, hover, zoom, and both Hurst heatmaps."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print('Plotly not installed: skipped interactive HTML. Install from requirements.txt.', file=sys.stderr)
        return
    fig = make_subplots(
        rows=4, cols=2, vertical_spacing=.082, horizontal_spacing=.09,
        row_heights=[.30, .30, .15, .25],
        specs=[[{'colspan':2},None],[{'colspan':2},None],[{'colspan':2},None],[{},{}]],
        subplot_titles=[
            '01 · Preço × tempo cronológico',
            '02 · Preço × tempo informacional',
            'Velocidade do relógio (dτ/dt)',
            '03 · Hurst: tempo cronológico', '03 · Hurst: tempo informacional'
        ]
    )
    fig.add_trace(go.Scattergl(
        x=df.timestamp, y=df.close, mode='lines', name='P(t)',
        line=dict(color='#2563eb',width=1.5),
        customdata=df.tau_minutes,
        hovertemplate='Data UTC: %{x}<br>Preço: %{y:,.2f}<br>τ: %{customdata:,.2f}<extra></extra>'
    ),row=1,col=1)
    for idx, cand in enumerate(candidates[:2]):
        indices=cand['pivot_indices']
        fig.add_trace(go.Scatter(
            x=df.timestamp.iloc[indices],y=df.close.iloc[indices],mode='markers+lines',
            marker=dict(size=6,color='#d97706'),line=dict(color='#d97706',width=1),
            name=f'Impulso Elliott {idx+1} (score {cand["score"]:.2f})',
            hovertemplate='Extremo retrospectivo: %{x}<br>Preço: %{y:,.2f}<extra></extra>'
        ),row=1,col=1)
    fig.add_trace(go.Scattergl(
        x=df.tau_minutes,y=df.close,mode='lines',name='P(τ)',
        line=dict(color='#0f766e',width=1.5),
        customdata=np.array(df.timestamp.dt.strftime('%Y-%m-%d %H:%M UTC')),
        hovertemplate='τ: %{x:,.2f}<br>Preço: %{y:,.2f}<br>Data: %{customdata}<extra></extra>'
    ),row=2,col=1)
    fig.add_trace(go.Scattergl(
        x=df.timestamp,y=df.clock_speed,mode='lines',name='Velocidade',
        line=dict(color='#d97706',width=1),
        hovertemplate='Data UTC: %{x}<br>dτ/dt: %{y:.3f}<extra></extra>'
    ),row=3,col=1)
    marks=spec.drop_duplicates('end_idx').end_idx.to_numpy(dtype=int)
    labels=list(FIB_BANDS.keys())
    for col, variable in [(1,'H_calendar'),(2,'H_information')]:
        matrix=spec.pivot(index='scale',columns='end_idx',values=variable).reindex(labels).reindex(columns=marks)
        fig.add_trace(go.Heatmap(
            x=df.timestamp.iloc[marks],y=labels,z=matrix.to_numpy(),
            zmin=.1,zmax=.9,colorscale='Viridis',
            showscale=(col==2),
            colorbar=dict(title='H',len=.24,y=.12) if col==2 else None,
            hovertemplate='Data: %{x}<br>Banda: %{y}<br>H: %{z:.3f}<extra></extra>'
        ),row=4,col=col)
    fig.update_layout(
        height=1140, width=1270, autosize=True, template='plotly_white',
        title=dict(text='Elliott–Fibonacci × Tempo Informacional | '+source,x=.5),
        legend=dict(orientation='h',yanchor='bottom',y=1.025,x=.02),
        margin=dict(l=85,r=50,t=140,b=65),hovermode='closest'
    )
    fig.update_yaxes(title_text='Preço (USDT)',row=1,col=1)
    fig.update_yaxes(title_text='Preço (USDT)',row=2,col=1)
    fig.update_xaxes(title_text='τ (minutos informacionais)',row=2,col=1)
    fig.update_yaxes(title_text='dτ/dt',row=3,col=1)
    fig.write_html(dest/'PAINEL_INTERATIVO.html',include_plotlyjs=True,full_html=True)


def summarize(df: pd.DataFrame, spectra: pd.DataFrame, candidates: list[dict], source: str) -> dict:
    means = spectra.groupby('scale',sort=False)[['H_calendar','H_information','R2_calendar','R2_information']].mean()
    return {
        'data': source,
        'start_utc': str(df.timestamp.iloc[0]), 'end_utc':str(df.timestamp.iloc[-1]),
        'bars':int(len(df)),
        'calendar_minutes':float(df.calendar_minutes.iloc[-1]),
        'information_minutes':float(df.tau_minutes.iloc[-1]),
        'speed_mean':float(df.clock_speed.mean()),
        'speed_min':float(df.clock_speed.min()),
        'speed_max':float(df.clock_speed.max()),
        'hursts_by_scale': {band:{metric:round(float(v),5) for metric,v in row.items()}
                            for band,row in means.to_dict('index').items()},
        'fib_impulse_candidates_detected_top5':candidates,
        'important_note':('Activity-derived informational clock; NOT direct news entropy. '
                          'Hurst local-fit R² is descriptive, not out-of-sample predictive performance. '
                          'Fibonacci impulse detection requires reversal confirmation.'),
    }


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--csv',type=Path,help='CSV with the 6 required columns')
    group.add_argument('--binance-month',help='Monthly archive period YYYY-MM (requires network)')
    group.add_argument('--binance-day',help='Daily archive period YYYY-MM-DD (requires network)')
    parser.add_argument('--symbol',default='BTCUSDT')
    parser.add_argument('--interval',default='1m')
    parser.add_argument('--out',type=Path,default=Path('resultados'))
    parser.add_argument('--max-bars',type=int,default=12000)
    parser.add_argument('--demo-bars',type=int,default=5000)
    parser.add_argument('--reversal',type=float,default=0.0025,help='Log-price reversal required to confirm a pivot (default 0.0025)')
    args=parser.parse_args(argv)
    args.out.mkdir(parents=True,exist_ok=True)
    if args.csv:
        df=standardize(pd.read_csv(args.csv));source='CSV de usuário: '+str(args.csv)
    elif args.binance_month:
        df=download_binance(args.symbol,args.interval,args.binance_month,'monthly')
        source=f'Binance SPOT {args.symbol} {args.interval} {args.binance_month}'
    elif args.binance_day:
        df=download_binance(args.symbol,args.interval,args.binance_day,'daily')
        source=f'Binance SPOT {args.symbol} {args.interval} {args.binance_day}'
    else:
        df=synthetic_data(n=args.demo_bars)
        source='SIMULAÇÃO SINTÉTICA (NÃO DADOS REAIS)'
    if args.max_bars < 750:
        raise ValueError('--max-bars must be at least 750')
    if len(df)>args.max_bars:
        df=df.iloc[-args.max_bars:].reset_index(drop=True)
    df=information_clock(df)
    spectra=hurst_spectra(df)
    logprice=np.log(df.close.to_numpy())
    pivots=confirmed_pivots(logprice, reversal=args.reversal)
    candidates=best_fibonacci_impulses(logprice,pivots)
    plot_outputs(df,spectra,candidates,args.out,source)
    write_interactive_dashboard(df,spectra,candidates,args.out,source)
    df.to_csv(args.out/'relogio_informacional.csv',index=False)
    spectra.to_csv(args.out/'hurst_por_escala.csv',index=False)
    stats=summarize(df,spectra,candidates,source)
    (args.out/'diagnosticos.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(stats,indent=2,ensure_ascii=False))
    print('\nFiles saved to:',args.out.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
