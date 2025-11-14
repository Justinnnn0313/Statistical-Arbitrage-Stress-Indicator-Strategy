import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, Dict, List


# ========================= 配置常量（最优参数） =========================
CONST = {
    'METH_PATH': 'D:/桌面/WEEK1/商品期货数据/day_20220611/R.CN.CZC.MA.0004.csv',
    'EG_PATH': 'D:/桌面/WEEK1/商品期货数据/day_20220611/R.CN.DCE.eg.0004.csv',
    'STRESS_WINDOW': 14,
    'ATR_WINDOW': 20,
    'CORR_WINDOW': 60,
    'MIN_CORRELATION': 0.4, 
    'ENTER_THRESH_HIGH': 95,
    'ENTER_THRESH_LOW': 5,
    'EXIT_THRESH_HIGH': 60,
    'EXIT_THRESH_LOW': 40,
    'ACCOUNT_CAPITAL': 1000000.0,
    'RISK_PCT': 0.02,
    'RISK_MULTIPLIER': 2,  
    'TIME_MULTIPLIER': 2,
    'LOSS_MULTIPLIER': 2,
    'CORR_STOP_THRESHOLD': 0.3,
    'CONTRACT_MULTIPLIER_MA': 10,
    'CONTRACT_MULTIPLIER_EG': 10,
    'MIN_TICK': 1.0,
    'SLIPPAGE_RATE': 0.0002,
    'SENS_RISK_LIST': [1,2,3,4,5],
    'SENS_ENTER_LIST': [90,92,94,95,96,98]
}


def load_csv_auto(path: str, date_col_candidates=('CLOCK','DATE','DATETIME','TIME')) -> Tuple[pd.DataFrame, str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    colmap = {c: c.upper() for c in df.columns}
    df = df.rename(columns=colmap)
    date_col_found = None
    for cand in date_col_candidates:
        if cand.upper() in df.columns:
            date_col_found = cand.upper()
            break
    if date_col_found is None:
        for c in df.columns:
            if 'TIME' in c or 'DATE' in c:
                date_col_found = c
                break
    if date_col_found is None:
        raise KeyError("无法识别日期列")
    df[date_col_found] = pd.to_datetime(df[date_col_found], errors='coerce')
    for c in df.columns:
        if c == date_col_found:
            continue
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.sort_values(date_col_found).reset_index(drop=True)
    return df, date_col_found

def prepare_data(meth_path: str, eg_path: str) -> pd.DataFrame:
    df_ma, date_col_ma = load_csv_auto(meth_path)
    df_eg, date_col_eg = load_csv_auto(eg_path)
    df_ma = df_ma.rename(columns={date_col_ma: 'DATE'})
    df_eg = df_eg.rename(columns={date_col_eg: 'DATE'})
    
    def add_suffix(df_in, suffix):
        df2 = df_in.copy()
        rename_map = {}
        for c in df2.columns:
            if c == 'DATE':
                continue
            rename_map[c] = f"{c}_{suffix}"
        df2 = df2.rename(columns=rename_map)
        return df2
    
    left = add_suffix(df_ma, 'MA')
    right = add_suffix(df_eg, 'EG')
    merged = pd.merge(left, right, on='DATE', how='inner')
    
    for suffix in ('MA','EG'):
        for base in ['CLOSE','OPEN','HIGH','LOW']:
            col = f"{base}_{suffix}"
            if col not in merged.columns:
                close_col = f"CLOSE_{suffix}"
                if close_col in merged.columns:
                    merged[col] = merged[close_col]
                else:
                    merged[col] = np.nan
    
    merged = merged.sort_values('DATE').reset_index(drop=True)
    merged = merged.dropna(subset=['CLOSE_MA', 'CLOSE_EG']).reset_index(drop=True)
    return merged

# ========================= 指标函数 =========================
def rolling_stochastic(close: pd.Series, high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    lowest = low.rolling(window, min_periods=1).min()
    highest = high.rolling(window, min_periods=1).max()
    denom = highest - lowest
    denom = denom.replace(0, np.nan)
    stoch = (close - lowest) / denom * 100.0
    return stoch

def compute_stress(df: pd.DataFrame, window: int) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    stoch1 = rolling_stochastic(df['CLOSE_MA'], df['HIGH_MA'], df['LOW_MA'], window)
    stoch2 = rolling_stochastic(df['CLOSE_EG'], df['HIGH_EG'], df['LOW_EG'], window)
    diff = stoch1 - stoch2
    diff_min = diff.rolling(window, min_periods=1).min()
    diff_max = diff.rolling(window, min_periods=1).max()
    denom = diff_max - diff_min
    denom = denom.replace(0, np.nan)
    stress_raw = (diff - diff_min) / denom * 100.0
    return stress_raw, stoch1, stoch2, diff

def compute_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, atr_window: int) -> pd.Series:
    tr = compute_true_range(high, low, close)
    atr = tr.rolling(window=atr_window, min_periods=atr_window).mean()
    return atr

def compute_rolling_correlation(close1: pd.Series, close2: pd.Series, window: int) -> pd.Series:
    rolling_corr = close1.rolling(window=window).corr(close2)
    return rolling_corr

# ========================= 信号与头寸 =========================
def generate_signals_and_sizes(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    stress_raw, stoch1, stoch2, diff = compute_stress(df, params['STRESS_WINDOW'])
    df['STRESS_RAW'] = stress_raw
    df['STRESS_SIGNAL'] = stress_raw.shift(1)
    df['STOCH1_SIGNAL'] = stoch1.shift(1)
    df['STOCH2_SIGNAL'] = stoch2.shift(1)
    df['DIFF_SIGNAL'] = diff.shift(1)
    
    rolling_corr = compute_rolling_correlation(
        df['CLOSE_MA'], 
        df['CLOSE_EG'], 
        params['CORR_WINDOW']
    )
    df['ROLLING_CORR'] = rolling_corr
    df['CORR_SIGNAL'] = rolling_corr.shift(1)
    
    corr_filter = df['CORR_SIGNAL'] >= params['MIN_CORRELATION']
    
    df['ENTER_SHORT1_LONG2'] = (df['STRESS_SIGNAL'] >= params['ENTER_THRESH_HIGH']) & corr_filter
    df['ENTER_LONG1_SHORT2'] = (df['STRESS_SIGNAL'] <= params['ENTER_THRESH_LOW']) & corr_filter
    
    atr_ma = compute_atr(df['HIGH_MA'], df['LOW_MA'], df['CLOSE_MA'], params['ATR_WINDOW'])
    atr_eg = compute_atr(df['HIGH_EG'], df['LOW_EG'], df['CLOSE_EG'], params['ATR_WINDOW'])
    df['ATR_MA'] = atr_ma
    df['ATR_EG'] = atr_eg
    
    base_risk = params['ACCOUNT_CAPITAL'] * params['RISK_PCT']
    fixed_risk = base_risk * params['RISK_MULTIPLIER']
    cm1 = params['CONTRACT_MULTIPLIER_MA']
    cm2 = params['CONTRACT_MULTIPLIER_EG']
    
    df['POSITION_SIZE_MA'] = (fixed_risk / (df['ATR_MA'] * cm1)).fillna(0).apply(np.floor).astype(int)
    df['POSITION_SIZE_EG'] = (fixed_risk / (df['ATR_EG'] * cm2)).fillna(0).apply(np.floor).astype(int)
    
    return df

# ========================= 回测 =========================
def backtest_simulation(df: pd.DataFrame, params: Dict) -> Tuple[pd.DataFrame, List[Dict]]:
    df = df.copy().reset_index(drop=True)
    balance = params['ACCOUNT_CAPITAL']
    base_risk = params['ACCOUNT_CAPITAL'] * params['RISK_PCT']
    fixed_risk = base_risk * params['RISK_MULTIPLIER']
    time_stop_period = params['STRESS_WINDOW'] * params['TIME_MULTIPLIER']
    loss_stop_threshold = fixed_risk * params['LOSS_MULTIPLIER']
    cm1 = params['CONTRACT_MULTIPLIER_MA']
    cm2 = params['CONTRACT_MULTIPLIER_EG']
    slippage_rate = params.get('SLIPPAGE_RATE', 0.0002)
    corr_stop_threshold = params.get('CORR_STOP_THRESHOLD', 0.0)  # 持仓期相关性止损阈值
    
    position = None
    entry_idx = None
    entry_prices = {}
    pos_sizes = {}
    directions = {}
    pnl_records = []
    trades = []
    skipped_signals = 0
    corr_stops = 0  # 相关性止损次数
    
    for idx, row in df.iterrows():
        date = row['DATE']
        close_ma = row['CLOSE_MA']
        close_eg = row['CLOSE_EG']
        size_ma = int(row.get('POSITION_SIZE_MA', 0))
        size_eg = int(row.get('POSITION_SIZE_EG', 0))
        exec_price_ma = close_ma
        exec_price_eg = close_eg
        stress_signal = row.get('STRESS_SIGNAL', np.nan)
        current_corr = row.get('ROLLING_CORR', np.nan)  # 获取当前相关性
        
        realized_today = 0.0
        
        if position is not None:
            pnl1 = (exec_price_ma - entry_prices['MA']) * pos_sizes['MA'] * cm1 * directions['MA']
            pnl2 = (exec_price_eg - entry_prices['EG']) * pos_sizes['EG'] * cm2 * directions['EG']
            floating_pnl = pnl1 + pnl2
            
            # 新增: 持仓期相关性止损（优先级最高）
            if corr_stop_threshold > 0 and not np.isnan(current_corr) and current_corr < corr_stop_threshold:
                exit_value = (exec_price_ma * pos_sizes['MA'] * cm1) + (exec_price_eg * pos_sizes['EG'] * cm2)
                slippage_exit = exit_value * slippage_rate
                realized = floating_pnl - slippage_exit
                balance += realized
                realized_today = realized
                corr_stops += 1
                trades.append({
                    'entry_idx': entry_idx, 'exit_idx': idx, 'position': position,
                    'exit_reason': 'CORR_STOP',
                    'entry_price_MA': entry_prices['MA'], 'entry_price_EG': entry_prices['EG'],
                    'exit_price_MA': exec_price_ma, 'exit_price_EG': exec_price_eg,
                    'size_MA': pos_sizes['MA'], 'size_EG': pos_sizes['EG'],
                    'pnl': realized, 'holding_days': idx - entry_idx
                })
                position, entry_idx, entry_prices, pos_sizes, directions = None, None, {}, {}, {}
            # 浮亏止损
            elif floating_pnl <= -abs(loss_stop_threshold):
                exit_value = (exec_price_ma * pos_sizes['MA'] * cm1) + (exec_price_eg * pos_sizes['EG'] * cm2)
                slippage_exit = exit_value * slippage_rate
                realized = floating_pnl - slippage_exit
                balance += realized
                realized_today = realized
                trades.append({
                    'entry_idx': entry_idx, 'exit_idx': idx, 'position': position,
                    'exit_reason': 'LOSS_STOP',
                    'entry_price_MA': entry_prices['MA'], 'entry_price_EG': entry_prices['EG'],
                    'exit_price_MA': exec_price_ma, 'exit_price_EG': exec_price_eg,
                    'size_MA': pos_sizes['MA'], 'size_EG': pos_sizes['EG'],
                    'pnl': realized, 'holding_days': idx - entry_idx
                })
                position, entry_idx, entry_prices, pos_sizes, directions = None, None, {}, {}, {}
            else:
                holding_days = idx - entry_idx if entry_idx is not None else 0
                if holding_days >= time_stop_period:
                    exit_value = (exec_price_ma * pos_sizes['MA'] * cm1) + (exec_price_eg * pos_sizes['EG'] * cm2)
                    slippage_exit = exit_value * slippage_rate
                    realized = floating_pnl - slippage_exit
                    balance += realized
                    realized_today = realized
                    trades.append({
                        'entry_idx': entry_idx, 'exit_idx': idx, 'position': position,
                        'exit_reason': 'TIME_STOP',
                        'entry_price_MA': entry_prices['MA'], 'entry_price_EG': entry_prices['EG'],
                        'exit_price_MA': exec_price_ma, 'exit_price_EG': exec_price_eg,
                        'size_MA': pos_sizes['MA'], 'size_EG': pos_sizes['EG'],
                        'pnl': realized, 'holding_days': holding_days
                    })
                    position, entry_idx, entry_prices, pos_sizes, directions = None, None, {}, {}, {}
        
        if position is None:
            corr_signal = row.get('CORR_SIGNAL', np.nan)
            if not np.isnan(stress_signal):
                if (stress_signal >= params['ENTER_THRESH_HIGH'] or stress_signal <= params['ENTER_THRESH_LOW']):
                    if np.isnan(corr_signal) or corr_signal < params['MIN_CORRELATION']:
                        skipped_signals += 1
            
            if row.get('ENTER_SHORT1_LONG2', False) and size_ma > 0 and size_eg > 0:
                position = 'SHORT_MA_LONG_EG'
                entry_idx, entry_prices = idx, {'MA': exec_price_ma, 'EG': exec_price_eg}
                pos_sizes, directions = {'MA': size_ma, 'EG': size_eg}, {'MA': -1, 'EG': +1}
                entry_value = (exec_price_ma * size_ma * cm1) + (exec_price_eg * size_eg * cm2)
                balance -= entry_value * slippage_rate
                trades.append({
                    'entry_idx': idx, 'position': position,
                    'entry_price_MA': exec_price_ma, 'entry_price_EG': exec_price_eg,
                    'size_MA': size_ma, 'size_EG': size_eg, 'slippage_entry': entry_value * slippage_rate
                })
            elif row.get('ENTER_LONG1_SHORT2', False) and size_ma > 0 and size_eg > 0:
                position = 'LONG_MA_SHORT_EG'
                entry_idx, entry_prices = idx, {'MA': exec_price_ma, 'EG': exec_price_eg}
                pos_sizes, directions = {'MA': size_ma, 'EG': size_eg}, {'MA': +1, 'EG': -1}
                entry_value = (exec_price_ma * size_ma * cm1) + (exec_price_eg * size_eg * cm2)
                balance -= entry_value * slippage_rate
                trades.append({
                    'entry_idx': idx, 'position': position,
                    'entry_price_MA': exec_price_ma, 'entry_price_EG': exec_price_eg,
                    'size_MA': size_ma, 'size_EG': size_eg, 'slippage_entry': entry_value * slippage_rate
                })
        else:
            if position == 'SHORT_MA_LONG_EG' and not np.isnan(stress_signal) and stress_signal <= params['EXIT_THRESH_HIGH']:
                pnl1 = (exec_price_ma - entry_prices['MA']) * pos_sizes['MA'] * cm1 * directions['MA']
                pnl2 = (exec_price_eg - entry_prices['EG']) * pos_sizes['EG'] * cm2 * directions['EG']
                exit_value = (exec_price_ma * pos_sizes['MA'] * cm1) + (exec_price_eg * pos_sizes['EG'] * cm2)
                realized = pnl1 + pnl2 - exit_value * slippage_rate
                balance += realized
                realized_today = realized
                trades.append({
                    'entry_idx': entry_idx, 'exit_idx': idx, 'position': position,
                    'exit_reason': 'EXIT_BY_STRESS',
                    'entry_price_MA': entry_prices['MA'], 'entry_price_EG': entry_prices['EG'],
                    'exit_price_MA': exec_price_ma, 'exit_price_EG': exec_price_eg,
                    'size_MA': pos_sizes['MA'], 'size_EG': pos_sizes['EG'],
                    'pnl': realized, 'holding_days': idx - entry_idx
                })
                position, entry_idx, entry_prices, pos_sizes, directions = None, None, {}, {}, {}
            elif position == 'LONG_MA_SHORT_EG' and not np.isnan(stress_signal) and stress_signal >= params['EXIT_THRESH_LOW']:
                pnl1 = (exec_price_ma - entry_prices['MA']) * pos_sizes['MA'] * cm1 * directions['MA']
                pnl2 = (exec_price_eg - entry_prices['EG']) * pos_sizes['EG'] * cm2 * directions['EG']
                exit_value = (exec_price_ma * pos_sizes['MA'] * cm1) + (exec_price_eg * pos_sizes['EG'] * cm2)
                realized = pnl1 + pnl2 - exit_value * slippage_rate
                balance += realized
                realized_today = realized
                trades.append({
                    'entry_idx': entry_idx, 'exit_idx': idx, 'position': position,
                    'exit_reason': 'EXIT_BY_STRESS',
                    'entry_price_MA': entry_prices['MA'], 'entry_price_EG': entry_prices['EG'],
                    'exit_price_MA': exec_price_ma, 'exit_price_EG': exec_price_eg,
                    'size_MA': pos_sizes['MA'], 'size_EG': pos_sizes['EG'],
                    'pnl': realized, 'holding_days': idx - entry_idx
                })
                position, entry_idx, entry_prices, pos_sizes, directions = None, None, {}, {}, {}
        
        if position is not None:
            pnl1 = (exec_price_ma - entry_prices['MA']) * pos_sizes['MA'] * cm1 * directions['MA']
            pnl2 = (exec_price_eg - entry_prices['EG']) * pos_sizes['EG'] * cm2 * directions['EG']
            inter_pnl = pnl1 + pnl2
        else:
            inter_pnl = 0.0
        
        pnl_records.append({
            'time': date, 'realizedPNL': realized_today, 'interPNL': inter_pnl,
            'balance': balance, 'equity': balance + inter_pnl
        })
    
    result_df = pd.DataFrame(pnl_records)
    print(f"\n相关性过滤统计: 共跳过 {skipped_signals} 个信号 (相关性 < {params['MIN_CORRELATION']})")
    if corr_stop_threshold > 0:
        print(f"持仓期相关性止损: 触发 {corr_stops} 次 (相关性 < {corr_stop_threshold})")
    return result_df, trades

# ========================= 绩效统计 =========================
def analyze_trades(trades: List[Dict]) -> Dict:
    if not trades:
        return {}
    tdf = pd.DataFrame(trades)
    trades_done = tdf[tdf['pnl'].notna()].copy()
    trades_done['pnl'] = pd.to_numeric(trades_done['pnl'], errors='coerce')
    trades_done['holding_days'] = trades_done['holding_days'].astype(int)
    
    total_trades = len(trades_done)
    total_realized = trades_done['pnl'].sum()
    wins = trades_done[trades_done['pnl'] > 0]
    losses = trades_done[trades_done['pnl'] <= 0]
    win_rate = len(wins) / total_trades if total_trades>0 else np.nan
    avg_pnl = trades_done['pnl'].mean()
    avg_win = wins['pnl'].mean() if len(wins)>0 else 0.0
    avg_loss = losses['pnl'].mean() if len(losses)>0 else 0.0
    avg_holding = trades_done['holding_days'].mean()
    largest_win = trades_done['pnl'].max()
    largest_loss = trades_done['pnl'].min()
    expectancy = (win_rate * (avg_win if not np.isnan(avg_win) else 0) + (1-win_rate) * (avg_loss if not np.isnan(avg_loss) else 0))
    
    return {
        'total_trades': total_trades,
        'total_realized_pnl': total_realized,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_holding_days': avg_holding,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'expectancy_per_trade': expectancy,
        'trades_df_head': trades_done.head(10)
    }

def performance_metrics(pnl_df: pd.DataFrame, init_capital: float) -> Dict:
    df = pnl_df.copy().sort_values('time').reset_index(drop=True)
    df['daily_return'] = df['equity'].pct_change().fillna(0)
    days = (df['time'].iloc[-1] - df['time'].iloc[0]).days if len(df) > 1 else 1
    cumulative_return = df['equity'].iloc[-1] / (init_capital) - 1.0
    annual_return = (1.0 + cumulative_return) ** (365.0 / days) - 1.0 if days > 0 else 0.0
    trading_days = 252.0
    annual_vol = df['daily_return'].std() * math.sqrt(trading_days) if df['daily_return'].std() > 0 else 0.0
    sharpe = (df['daily_return'].mean() * trading_days) / annual_vol if annual_vol > 0 else np.nan
    cummax = df['equity'].cummax()
    drawdown = (df['equity'] - cummax) / cummax
    max_drawdown = drawdown.min()
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
    return {
        'Begin': df['time'].iloc[0],
        'End': df['time'].iloc[-1],
        'InitCap': init_capital,
        'AnnualReturn': annual_return,
        'CumulativeReturn': cumulative_return,
        'StdDev': annual_vol,
        'Sharpe': sharpe,
        'Calmar': calmar,
        'MaxDrawdown': max_drawdown
    }

# ========================= 可视化 =========================
def plot_equity_and_drawdown(pnl_df: pd.DataFrame):
    df = pnl_df.copy().sort_values('time')
    fig, ax = plt.subplots(2,1, figsize=(12,8), sharex=True, gridspec_kw={'height_ratios':[3,1]})
    ax[0].plot(df['time'], df['equity'], label='Equity (balance + interPNL)')
    ax[0].plot(df['time'], df['balance'], label='Balance (realized only)', linestyle='--')
    ax[0].set_title('Equity & Balance')
    ax[0].legend()
    cummax = df['equity'].cummax()
    drawdown = (df['equity'] - cummax) / cummax
    ax[1].plot(df['time'], drawdown, label='Drawdown', color='red')
    ax[1].fill_between(df['time'], drawdown, 0, where=(drawdown<0), color='red', alpha=0.2)
    ax[1].set_title('Drawdown')
    plt.tight_layout()
    plt.savefig('D:/桌面/WEEK1/equity_drawdown.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图表已保存: D:/桌面/WEEK1/equity_drawdown.png")

def plot_stress_signals_and_markers(df: pd.DataFrame, params: Dict, trades: List[Dict]):
    plt.figure(figsize=(14,6))
    ax = plt.subplot(211)
    ax.plot(df['DATE'], df['CLOSE_MA'], label='MA Price', linewidth=1)
    ax.plot(df['DATE'], df['CLOSE_EG'], label='EG Price', linewidth=1)
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    
    ax2 = plt.subplot(212, sharex=ax)
    ax2.plot(df['DATE'], df['STRESS_RAW'], label='Stress', linewidth=1)
    ax2.axhline(params['ENTER_THRESH_HIGH'], linestyle='--', label='Enter High')
    ax2.axhline(params['ENTER_THRESH_LOW'], linestyle='--', label='Enter Low')
    ax2.axhline(params['EXIT_THRESH_HIGH'], linestyle=':', label='Exit High')
    ax2.axhline(params['EXIT_THRESH_LOW'], linestyle=':', label='Exit Low')
    for tr in trades:
        eidx = tr.get('entry_idx')
        xidx = tr.get('exit_idx', None)
        if eidx is not None and eidx < len(df):
            ax2.axvline(df.loc[eidx,'DATE'], color='green', alpha=0.3)
        if xidx is not None and xidx < len(df):
            ax2.axvline(df.loc[xidx,'DATE'], color='red', alpha=0.3)
    ax2.set_ylabel('Stress')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('D:/桌面/WEEK1/stress_signals.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图表已保存: D:/桌面/WEEK1/stress_signals.png")

def plot_positions(df: pd.DataFrame, trades: List[Dict]):
    plt.figure(figsize=(14,4))
    plt.plot(df['DATE'], df['CLOSE_MA'], label='MA Price')
    # 橙色区域表示持仓期间（从入场到出场）
    for tr in trades:
        entry_idx = tr.get('entry_idx')
        exit_idx = tr.get('exit_idx', entry_idx)
        if entry_idx is not None and entry_idx < len(df):
            entry_date = df.loc[entry_idx,'DATE']
            exit_date = df.loc[exit_idx,'DATE'] if (exit_idx is not None and exit_idx < len(df)) else df['DATE'].iloc[-1]
            plt.axvspan(entry_date, exit_date, alpha=0.12, color='orange')
    plt.title('持仓区间图 (橙色区域 = 持仓期间)')
    plt.xlabel('日期')
    plt.ylabel('MA价格')
    plt.legend()
    plt.tight_layout()
    plt.savefig('D:/桌面/WEEK1/positions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图表已保存: D:/桌面/WEEK1/positions.png")

def plot_trade_pnl_hist(trades: List[Dict]):
    if not trades:
        print("No trades to plot.")
        return
    tdf = pd.DataFrame(trades)
    if 'pnl' not in tdf.columns:
        print("No realized trades yet.")
        return
    plt.figure(figsize=(8,4))
    plt.hist(tdf['pnl'].dropna(), bins=30, edgecolor='k')
    plt.title('Distribution of Trade PnL')
    plt.xlabel('PnL per trade')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('D:/桌面/WEEK1/trade_pnl_hist.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图表已保存: D:/桌面/WEEK1/trade_pnl_hist.png")

def sensitivity_analysis_and_plot(base_df: pd.DataFrame, base_params: Dict, risk_mult_list=None, enter_thresh_list=None) -> pd.DataFrame:
    if risk_mult_list is None:
        risk_mult_list = base_params.get('SENS_RISK_LIST', [1,2,3,4,5])
    if enter_thresh_list is None:
        enter_thresh_list = base_params.get('SENS_ENTER_LIST', [90,92,94,95,96,98])
    results = []
    for rm in risk_mult_list:
        for et in enter_thresh_list:
            params = base_params.copy()
            params['RISK_MULTIPLIER'] = rm
            params['ENTER_THRESH_HIGH'] = et
            df = generate_signals_and_sizes(base_df.copy(), params)
            pnl_df, trades = backtest_simulation(df, params)
            perf = performance_metrics(pnl_df, params['ACCOUNT_CAPITAL'])
            results.append({
                'risk_multiplier': rm,
                'enter_thresh_high': et,
                'annual_return': perf['AnnualReturn'],
                'sharpe': perf['Sharpe'],
                'max_drawdown': perf['MaxDrawdown'],
                'cumulative_return': perf['CumulativeReturn']
            })
    sens_df = pd.DataFrame(results)
    pivot = sens_df.pivot(index='risk_multiplier', columns='enter_thresh_high', values='annual_return')
    plt.figure(figsize=(8,6))
    plt.imshow(pivot.values, aspect='auto', origin='lower')
    plt.colorbar(label='Annual Return')
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel('Enter Threshold High')
    plt.ylabel('Risk Multiplier')
    plt.title('Sensitivity: Annual Return')
    plt.tight_layout()
    plt.savefig('D:/桌面/WEEK1/sensitivity_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图表已保存: D:/桌面/WEEK1/sensitivity_analysis.png")
    return sens_df

# ========================= 主流程 =========================
def run_full_pipeline(meth_path=None, eg_path=None, params_override=None, run_sensitivity=False):
    params = CONST.copy()
    if params_override:
        params.update(params_override)
    meth = params['METH_PATH'] if meth_path is None else meth_path
    eg = params['EG_PATH'] if eg_path is None else eg_path
    
    print("Loading data...")
    df = prepare_data(meth, eg)
    print(f"Data loaded. Rows after alignment: {len(df)}")
    
    df = generate_signals_and_sizes(df, params)
    pnl_df, trades = backtest_simulation(df, params)
    perf = performance_metrics(pnl_df, params['ACCOUNT_CAPITAL'])
    trade_stats = analyze_trades(trades)
    
    print("\nBacktest finished. Performance summary:")
    for k, v in perf.items():
        print(f"{k}: {v}")
    
    print("\nTrade-level stats:")
    for k,v in trade_stats.items():
        if k != 'trades_df_head':
            print(f"{k}: {v}")
    if 'trades_df_head' in trade_stats:
        print("\nExample trades:")
        print(trade_stats['trades_df_head'])
    
    print("\nDisplaying Equity & Drawdown...")
    plot_equity_and_drawdown(pnl_df)
    print("Displaying Stress & Signals (entry=green line, exit=red line)...")
    plot_stress_signals_and_markers(df, params, trades)
    print("Displaying Positions highlight...")
    plot_positions(df, trades)
    print("Displaying Trade PnL histogram...")
    plot_trade_pnl_hist(trades)
    
    if run_sensitivity:
        print("Running sensitivity analysis (this may take time)...")
        sens_df = sensitivity_analysis_and_plot(df, params)
    else:
        sens_df = pd.DataFrame()
    
    return pnl_df, perf, trades, sens_df

if __name__ == '__main__':
    pnl_df, perf, trades, sens_df = run_full_pipeline(run_sensitivity=False)