# Statistical Arbitrage Strategy with the Stress Indicator

## Overview

This repository contains a complete Python implementation of a **statistical arbitrage** strategy based on the **Stress Indicator**, a concept from Perry Kaufman's book, *Trading Systems and Methods*. The strategy is designed to capitalize on the mean-reverting behavior of the spread between two highly correlated assets. This implementation backtests the strategy on the commodity futures pair: Methanol (MA) and Ethylene Glycol (EG).

The core of the strategy is a "triple stochastic" process that quantifies the temporary dislocations in the price relationship between the two assets. By identifying statistically significant deviations from the norm, the strategy enters trades to profit from the expected convergence back to the historical mean.

This project provides a fully vectorized backtesting engine, from data loading and signal generation to trade simulation and performance analysis.

## Key Features

- **Data Handling**: Automatically loads and aligns time-series data for the asset pair.
- **Indicator Calculation**: Implements the full logic for the Stress Indicator, which normalizes the spread's relative strength into a bounded oscillator (0-100).
- **Risk-Equalized Position Sizing**: Uses Average True Range (ATR) to calculate position sizes, ensuring equal risk contribution from each leg of the trade.
- **Advanced Risk Management**:
    - **Entry Filter**: A rolling correlation filter to ensure the statistical relationship is stable before initiating a trade.
    - **Profit-Taking Exit**: Exits trades when the Stress indicator reverts to a neutral (mean) zone.
    - **Time Stop**: A structural stop-loss that closes positions held longer than a predefined period.
    - **Loss Stop**: A hard stop-loss based on a multiple of the initial trade risk to protect against model failure.
    - **Correlation Stop**: A dynamic stop-loss that exits an open position if the correlation between the assets deteriorates, indicating a potential breakdown of the statistical relationship.
- **Comprehensive Performance Analytics**: Calculates key metrics such as Annual Return, Sharpe Ratio, Maximum Drawdown, and Calmar Ratio. It also provides detailed trade-level statistics.
- **Visualization**: Generates plots for the equity curve, drawdown, stress indicator with trade markers, and position holding periods.
- **Sensitivity Analysis**: Includes a module to test strategy robustness across a range of key parameters.

## Strategy Logic

The strategy is built on a three-step process to generate the Stress Indicator, which models the spread's behavior:

1.  **Calculate Individual Stochastics**: A standard stochastic oscillator is calculated for both Methanol (Asset 1) and Ethylene Glycol (Asset 2) over a 14-day window.
2.  **Calculate the Difference**: A new time series is created by subtracting the stochastic of Asset 2 from the stochastic of Asset 1 (`Diff = Stoch1 - Stoch2`). This series represents the raw, unnormalized spread.
3.  **Calculate Stress**: A final stochastic calculation is applied to the `Diff` series itself. This normalizes the spread's movement into a 0-100 oscillator, which is the final **Stress Indicator**.

### Trading Rules

-   **Entry Signal (Divergence)**:
    -   Enter a **short spread** position (Short MA / Long EG) when the Stress Indicator crosses above 95, indicating the spread is at a statistical extreme.
    -   Enter a **long spread** position (Long MA / Short EG) when the Stress Indicator crosses below 5.
    -   An entry is only executed if the 60-day rolling correlation between the assets is above 0.4.
-   **Exit Signal (Mean Reversion)**:
    -   Exit a short spread position when the Stress Indicator reverts to 60.
    -   Exit a long spread position when the Stress Indicator reverts to 40.
-   **Stop-Losses**:
    -   **Time Stop**: Exit any position held for more than 28 days (`14-day window * 2`).
    -   **Loss Stop**: Exit if the floating loss exceeds 2x the initial risk amount.
    -   **Correlation Stop**: Exit if the 60-day correlation drops below 0.3 during an active trade.

## Backtest Performance Summary

-   **Annual Return**: 3.48%
-   **Cumulative Return**: 12.71%
-   **Sharpe Ratio**: 0.27
-   **Maximum Drawdown**: -23.28%
-   **Calmar Ratio**: 0.15
-   **Total Trades**: 70
-   **Win Rate**: 58.6%

*Note: These results are based on the specific dataset and parameters defined in the code. Past performance is not indicative of future results.*

## How to Use

### 1. Prerequisites

Ensure you have Python 3.x and the following libraries installed:

```bash
pip install pandas numpy matplotlib
```

### 2. Configuration

All strategy parameters, file paths, and settings are located in the `CONST` dictionary at the top of the script.

```python
CONST = {
    # --- IMPORTANT: Update these file paths ---
    'METH_PATH': 'path/to/your/methanol_data.csv',
    'EG_PATH': 'path/to/your/ethylene_glycol_data.csv',

    # --- Strategy Parameters ---
    'STRESS_WINDOW': 14,
    'ATR_WINDOW': 20,
    'CORR_WINDOW': 60,
    'MIN_CORRELATION': 0.4,
    'ENTER_THRESH_HIGH': 95,
    'ENTER_THRESH_LOW': 5,
    # ... and other parameters
}
```

**Before running, you must update `METH_PATH` and `EG_PATH` to point to your local data files.**

### 3. Running the Backtest

To run the full backtest pipeline, simply execute the Python script from your terminal (e.g., if your file is named `backtest.py`):

```bash
python backtest.py
```

The script will:
1.  Load and process the data.
2.  Run the backtest simulation.
3.  Print a detailed performance summary and trade statistics to the console.
4.  Save several plots (e.g., `equity_drawdown.png`, `stress_signals.png`) to the specified output directory.

### 4. Running Sensitivity Analysis

To run the sensitivity analysis, change the `run_sensitivity` flag to `True` in the main execution block at the bottom of the script:

```python
if __name__ == '__main__':
    pnl_df, perf, trades, sens_df = run_full_pipeline(run_sensitivity=True)
```

This will generate and save an additional plot, `sensitivity_analysis.png`, which shows a heatmap of annual returns across different parameter combinations.

## Code Structure

The script is organized into functional blocks:

-   `CONST`: A dictionary holding all configurable parameters.
-   **Data Loading & Preparation**: Functions `load_csv_auto` and `prepare_data`.
-   **Indicator Functions**: Functions for calculating Stochastics, Stress, ATR, and Correlation.
-   **Signals & Sizing**: `generate_signals_and_sizes` function to create trade signals and calculate position sizes.
-   **Backtest Simulation**: The main `backtest_simulation` loop that processes trades and manages P&L.
-   **Performance Analytics**: `analyze_trades` and `performance_metrics` for calculating statistics.
-   **Visualization**: A set of `plot_*` functions for generating charts.
-   **Main Pipeline**: The `run_full_pipeline` function orchestrates the entire process.

## License

This project is open-source and available under the [MIT License](LICENSE).

