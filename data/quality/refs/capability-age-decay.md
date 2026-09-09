# Capability age decay (size-prior fallback)

Used only when `Q_file` is missing. Measured benches are **not** age-discounted.

## Form

\[
\text{decay}(t) = 2^{-t / T_{1/2}} = \exp(-\ln 2 \cdot t / T_{1/2})
\]

with \(t\) = years since `pushed_at`, \(T_{1/2} = 1.10\) years.

Proxy capability:

`capability = size_prior(active_params, quant) × 0.45 × decay(t)`

If `pushed_at` is missing, `decay = 1` (size prior only).

## Why exponential / why 1.10y

1. **Published exponential half-life in the LLM market** — arXiv:2603.28576 (*Tiered Super-Moore’s Law*) fits economy-tier token prices as \(P(t) \propto e^{-0.629 t}\) (years), giving \(t_{1/2} = 1.10\) y. That is the clearest exponential decay constant in recent LLM literature; we reuse it as a calendar depreciation rate for *relative* standing of unmatched open weights.

2. **Capability is largely a date trend** — arXiv:2608.29420 finds a single factor explains ~74.5% of leaderboard variance and tracks release date (\(R^2 \approx 0.505\)); most of the gap between models months apart is calendar, not residual scale.

3. **Order-of-magnitude check** — Epoch’s open-vs-closed compute lag is ~1 year (range ~5–22 months). Epoch ECI frontier advances ~14 pts/year post-reasoning models — fixed older releases fall behind on a similar yearly cadence. Scientific adoption lifespan also compresses (~23%/year later release; arXiv:2604.07530).

## Not used

- **7-week frontier churn** (median time at #1) — too aggressive for local/open catalog ranking.
- **Pretraining Decay Half-Life (PDHL)** — knowledge staleness vs training cutoff; domain-specific (news months, encyclopedic years), not relative intelligence vs peers.
- **Deprecating measured `Q_file`** — benches already encode absolute performance; age decay applies only to the unmatched size prior.

Accessed: 2026-09-09.
