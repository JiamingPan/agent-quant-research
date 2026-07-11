# Event Study Interview Drill

Use this as a 10-minute closed-book drill. Answer each question aloud before reading the
model answer. Memorize the reasoning, not the wording.

## The Workflow

`run_event_study(ticker, event_date, window)` does this:

1. Load the complete internal price range through `get_price_data`.
2. Reduce intraday bars to one closing price per trading day.
3. Compute daily log returns in basis points:

   ```text
   return_t = log(close_t / close_(t-1)) * 10,000
   ```

4. Locate the event date in the trading-day series and select `[-window, +window]`.
5. Build the baseline from returns strictly before the first day of that event window.
6. Estimate expected return as the mean baseline log return.
7. Calculate daily abnormal return and cumulative abnormal return:

   ```text
   AR_t  = actual_return_t - expected_return
   CAR_t = sum of AR from the start of the event window through day t
   ```

8. Build a 95% CAR interval from 1,000 resamples of centered pre-event returns.
9. Return daily AR/CAR, the interval, `n_pre_obs`, and the leakage-check status.

The key boundary is:

```text
baseline dates < first date in the event window
event-window dates are never baseline dates
```

## The 10-Minute Quiz

### Minute 0-1: Explain the workflow

Without looking above, trace one request from price loading to the JSON response. State
where log returns, expected return, AR, CAR, and the confidence interval are calculated.

### Minute 1-4: Why must the baseline use only pre-event data?

Answer these aloud:

1. What is the baseline trying to estimate?
2. What happens if an event-window or post-event return enters it?
3. Why is the cutoff before the entire event window, not merely before `event_date`?
4. How does the implementation enforce the rule?

**Model answer:**

The baseline estimates the counterfactual return I would expect if the event had not
occurred. It must use only information available before the event window. If event or
post-event returns enter the baseline, the event's price effect contaminates expected
return. Part of the effect is then subtracted from itself, usually biasing abnormal return
toward zero. It is also look-ahead leakage because future information helped define the
counterfactual.

The cutoff is before day `-window`, not just before day `0`, because days such as `-1` may
already contain anticipation, information leakage, or positioning related to the event.
The code selects baseline dates strictly before the first event-window observation and
then calls an explicit assertion that fails if any baseline date is on or after that
cutoff.

**Short interview version:**

> The baseline is my no-event counterfactual. Using event or post-event observations would
> leak future information and absorb part of the event effect into expected return. I fit
> it strictly before the entire event window and enforce that boundary with an assertion.

### Minute 4-7: Why bootstrap instead of a basic t-test?

Answer these aloud:

1. What assumptions make a basic t-test questionable for financial returns?
2. What does resampling give us?
3. Does the current bootstrap solve autocorrelation?
4. What would you use if serial dependence were material?

**Model answer:**

A basic t-test relies on a standard-error model that is easiest to justify when errors are
independent and approximately normal. Financial returns can be heavy-tailed, skewed,
heteroskedastic, and serially dependent. A bootstrap uses the empirical pre-event residual
distribution, so it does not force a normal shape and can represent asymmetric or
heavy-tailed noise present in the sample.

However, the current MVP uses an **IID residual bootstrap**: it samples individual centered
pre-event returns independently. That helps with non-normality, but it does **not** preserve
autocorrelation. Never claim otherwise. If serial dependence matters, use a moving-block
or stationary bootstrap so adjacent returns are resampled together. A HAC/Newey-West
standard error is another defensible comparison.

The current choice is an auditable MVP, not a claim that IID bootstrap is universally
superior. Its deterministic seed also makes the API and tests reproducible.

**Short interview version:**

> I used an empirical bootstrap because returns need not be normally distributed. The MVP
> resamples centered pre-event residuals 1,000 times. It is IID, so it does not preserve
> autocorrelation; for production inference I would use a block bootstrap or compare
> against HAC errors.

### Minute 7-10: What does the confidence interval mean?

The implementation first centers the pre-event returns:

```text
residual_t = pre_event_return_t - mean(pre_event_returns)
```

For each of 1,000 bootstrap repetitions, it samples `event_n` residuals with replacement,
sums them, and adds that noise to observed CAR. The reported bounds are the 2.5th and 97.5th
percentiles of those simulated CAR values.

**Model answer:**

The code calls this a confidence interval; more precisely, it is an approximate bootstrap
uncertainty interval around observed CAR. It is valid only under specific assumptions: the
centered pre-event residuals must be representative of event-window noise, and the current
IID resampling assumes those residuals are exchangeable. Under repeated samples satisfying
those assumptions, this percentile procedure aims for approximately 95% coverage.

It does **not** mean there is a 95% probability that the event caused a return inside the
interval. It is not a causal claim, not a guarantee, and not a p-value. Because this
interval is centered on observed CAR, it also is not by itself a test of the null hypothesis
that event impact equals zero. A null test would construct a zero-centered bootstrap CAR
distribution and compare observed CAR with that distribution.

**Short interview version:**

> The interval is the range of CAR estimates produced by repeatedly injecting noise drawn
> from the pre-event residual distribution. It quantifies sampling uncertainty conditional
> on that historical noise being representative. It is not the probability that the event
> caused the move and it does not establish causality.

## Questions an Interviewer May Add

### Why log returns?

Log returns add across time, so cumulative return calculations are cleaner. They are also
approximately equal to simple returns for small moves. Multiplying by 10,000 expresses the
result in basis points.

### Why use trading-day positions rather than calendar days?

Markets do not trade every calendar day. For a Friday event, trading day `+1` is normally
Monday, not Saturday. The code locates the event inside the observed daily series and moves
by trading observations.

### Is `n_pre_obs >= 2` enough for trustworthy inference?

No. Two observations are only the code's minimum availability check. A bootstrap from two
returns is statistically weak. In a serious study, choose a much longer estimation window,
report its length, inspect stability, and run sensitivity checks.

### Does this prove the event caused CAR?

No. Other news, market moves, sector shocks, and confounders can occur in the same window.
This MVP measures return relative to a simple historical-mean baseline. Stronger designs
could add a market model, matched controls, narrower timestamps, and robustness checks.

## What Not to Claim

- Do not say the current IID bootstrap handles autocorrelation.
- Do not call the interval a 95% probability that the event caused the return.
- Do not call CAR causal evidence by itself.
- Do not say two baseline observations are statistically sufficient.
- Do not say post-event data are safe because they improve the sample size.

## The 60-Second Interview Answer

> I built the event study around a strict no-look-ahead boundary. I convert daily closes to
> log returns, locate the event window in trading-day space, and estimate expected return as
> the mean of observations strictly before day minus window. Daily abnormal return is actual
> minus expected, and CAR is its cumulative sum. An explicit assertion proves no event-window
> observation enters the baseline. For uncertainty, I generate 1,000 CAR draws from centered
> pre-event residuals and report the percentile interval. That avoids imposing a normal
> residual shape, but the MVP bootstrap is IID, so it does not preserve autocorrelation; a
> production version would use a block bootstrap or HAC comparison. The interval quantifies
> sampling uncertainty under those assumptions. It is not a causal probability or proof
> that the event caused the move.

## Cold-Pass Scorecard

You own this section when you can answer all five without notes:

- [ ] Draw the baseline and event-window timeline correctly.
- [ ] Explain leakage without using the phrase "because it is bad."
- [ ] Derive `AR_t` and `CAR_t` aloud.
- [ ] State exactly what IID bootstrap handles and does not handle.
- [ ] Interpret the interval without making a probability or causal claim.
