
# Credit card default prediction

### About the Project

This project predicts whether a credit card customer will default on their next payment. It uses advanced feature engineering (like debt acceleration, recency-weighted moving averages, and spending anomalies) to track a user's financial behavior over 6 months and feed those insights into a Machine Learning classification model.

## Exploratory data analysis (EDA)
A train test split was made early on and the EDA was only done on the train set to avoid information leakege.

[📊 View the full EDA report](https://mark-maged3.github.io/Credit_card_default_prediction/)

## Data Cleaning
43 duplicate rows were removed, and undocumented labels were replaced with nulls and imputated using MissForest

## Feature Engineering & Data Dictionary

### 1. Core Matrices Setup
Before creating features, chronological matrices were established for debt and payments. A baseline (0.5% of the credit limit) was introduced to prevent division-by-zero errors and scale metrics appropriately. A half-life decay of 2 months was set to prioritize recent behavior over older behavior.

### 2. Payoff Ratio Features
Evaluates how much of the actual debt the user is clearing, rather than just the raw payment amount.
* **`payoff_ratio_1` to `5`:** Ratio of lagged payments to the previous bill.
* **`mean_payoff_ratio`:** The 6-month average payoff ratio.
* **`full_payment_count`:** Count of months where the user paid >= 95% of their bill. *(Note: A 95% threshold is used instead of 100% to account for users paying their "Statement Balance" rather than their "Current Balance", mental rounding habits, and floating-point irregularities. Behaviorally, a 95%+ payment indicates full capacity to pay).*
* **`micro_payment_count` & `zero_payment_count`:** Tracks minimum payments (<= 10%) and missed payments.

### 3. Exponentially Weighted Moving Averages (EWMA)
Applies time-decay weights so that recent months impact the average more heavily than older months.
* **`ewma_payments` & `ewma_debt`:** Recency-weighted averages of payments and debts.
* **`ewma_payment_to_bill_ratios`:** A recency-weighted ratio of how well the user is covering their active debt.
* **`ewma_utilization`:** Recency-weighted average of credit utilization.

### 4. Debt Momentum Features
Treats debt as a trajectory to measure how fast it is growing or shrinking over time.
* **`debt_momentem`:** Uses Theil-Sen robust slopes to calculate normalized momentum, ignoring extreme outliers.
* **`ewls_debt_momentum`:** Calculates debt momentum using Exponentially Weighted Least Squares (EWLS), weighting recent trajectory shifts more heavily.

### 5. Debt Acceleration Features
Calculates the second derivative of a user's debt trajectory to measure if debt growth is speeding up.
* **`debt_accel_positive_count`:** How many times the user's debt growth accelerated.
* **`norm_debt_acceleration_mean`:** The average acceleration normalized against their total credit limit.

### 6. Debt Spike & Anomaly Detection
* **`debt_spike_ratio`:** Ratio of recent peak debt against historical minimum debt.
* **`debt_anomaly_score`:** Uses Median Absolute Deviation (MAD) to detect sudden, abnormal spikes in a user's current debt compared to their standard baseline.

### 7. Payment Volatility Features
* **`payment_volatility` & `relative_payment_volatility`:** Standard deviation of a user's payments (normalized).
* **`recent_payment_momentum`:** The first derivative of chronological payments to see if payment sizes are actively trending up or down.

### 8. Credit Utilization Features
* **`max_credit_utilization`:** peak percentage of the credit limit used.
* **`ewls_utilization_burn_rate`:** How fast the user is actively eating into their credit limit.
* **`recent_remaining_liquidity`:** Exact dollar amount of credit available in the most recent month.
* **`utilization_shock_recent`:** Sudden jumps in utilization.

### 9. Delinquency Features
Deep-dive into the user's late payment (PAY_X) statuses.
* **`ewma_pay_status` & `max_delinquency`:** Recency-weighted and maximum late statuses.
* **`delinquency_severity` & `delinquency_frequency`:** The total sum and count of late payment months.
* **`months_since_last_delinquency`:** Tracks how recently the user was late (6 = no recent delinquencies).
* **`max_delinquency_escalation`:** Tracks the maximum jump in delinquency (e.g., escalating from 1 month late to 3 months late).

### 10. Age-Related Features
Captures risk associated with demographics and life stages.
* **`limit_to_age_ratio` & `age_adjusted_utilization`:** Normalizes credit limits and EWMA utilization against the user's age.

### 11. New Spend Features
The raw dataset only provides "Bill Amount" (a mix of previous debt and new purchases). This section reverse-engineers the **actual new monthly spending**.
* **`new_spends1` to `5` & `total_new_spend`:** Extracting actual new purchases made each month.
* **`ewma_spends` & `ewls_spend_momentum`:** Recency-weighted average of new spends and the trajectory/momentum of their spending habits.
* **`total_spend_to_limit_ratio`:** Total new spend over the period divided by credit limit.

### 12. Activity & Variation Features
Tracks general account usage and inactivity.
* **`active_debt_months` & `active_spend_months`:** Count of months the user carried debt or made purchases.
* **`consecutive_zero_payments`:** Longest streak of no payments.
* **`debt_no_variation` & `spend_no_variation`:** Binary flags indicating zero variation in debt/spend (MAD = 0).
* **`spend_anomaly_score`:** MAD-based score to detect out-of-character spending sprees.
* **`spend_spike_ratio`:** Ratio of recent peak spending against historical minimum spending. Unlike the Debt Spike Ratio (which can trigger purely from compounding interest and missed payments), this feature specifically isolates active consumer behavior, flagging users who suddenly go on a massive, out-of-character spending spree.

### 13. Entropy Features (Predictability)
Measures the randomness of a user's financial life. High entropy indicates erratic behavior; low entropy indicates routine.
* **`PAY_STATUS_ENTROPY` (Shannon entropy of state changes):** Measures how chaotic a user's delinquency statuses are.
* **`payment_entropy` & `debt_entropy` (Permutation):** Time-series complexity metrics detecting if a user has a stable financial routine or highly chaotic fluctuations.

### 14. Pay Status Reversal Features
* **`pay_status_reversals`:** Tracks the number of times a user flips back and forth between paying on time and falling behind (a strong indicator of financial distress).

### 15. Behavioral Shift Features
Compares recent behavior (last 2 months) against historical baseline behavior (months 3-6).
* **`payoff_behavior_shift`:** Did the user suddenly stop paying off their balances?
* **`spend_behavior_shift`:** Did the user suddenly start spending much more or much less than usual?

##  Advanced Mathematical Concepts Used

To capture complex human financial behaviors, this project implements several advanced mathematical formulas from scratch using vectorized NumPy operations:

* **Exponentially Weighted Least Squares (EWLS):** Used to calculate debt and spend momentum. Instead of standard linear regression, EWLS calculates a line of best fit where recent months have a higher "gravitational pull" (using an exponential decay half-life of 2 months). This captures a user's current trajectory much better than an unweighted slope.

* **Theil-Sen Estimator:** Used as a secondary robust momentum metric. It computes the slopes of all possible pairs of time points across the 6 months and takes the median. This makes the momentum feature immune to single-month outliers or data glitches.

* **Median Absolute Deviation (MAD):** Used for Anomaly Detection (Debt Spikes and Spend Shocks). Because financial data is highly skewed, standard Z-scores fail. MAD replaces Mean/Standard Deviation with Median/Median-Deviation, creating a robust anomaly score to flag erratic behavior.

* **Transition Entropy:** Measures the diversity of state changes in a user's repayment status. Rather than treating the history as an unordered bag of values, it operates on first differences, which captures how a user moves between states. A person steadily worsening has low transition entropy, while
a person oscillating erratically has high transition entropy.

* **Weighted Permutation Entropy:** Calculates the time-series complexity of a
user's payment and debt histories using sliding windows of size 3. Unlike standard Permutation Entropy, which treats all ordinal patterns equally, the weighted variant scales each pattern's contribution by the amplitude variance within its window, so large swings dominate over noise-level fluctuations.
Note: In dynamical systems, entropy typically requires large time-series datasets to converge. Because we only have 6 months of data, this feature acts as a local complexity heuristic rather than a true systemic entropy measure.
