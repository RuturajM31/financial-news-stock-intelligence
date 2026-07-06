# Market Data Foundation Contract — Final Qualified Path

## Purpose

This package creates the verified evidence required before the stock-movement
model may train. It uses only the provider path already proven on the target
Mac:

- SEC EDGAR for official company-disclosure events;
- Tiingo EOD for authenticated, internal-use adjusted daily prices;
- the existing local DistilBERT deployment champion for sentiment inference.

It does not use GDELT, Nasdaq website endpoints, Stooq, or unauthenticated price
scraping. It does not train the movement model and does not change deployment.

## Fixed experiment window

- Tiingo price window: 2015-01-01 through 2020-04-01.
- SEC event window: 2015-02-02 through 2020-03-31.
- The earlier price start supplies prior-session and rolling-feature context.
- The later price end supplies a future target session for end-window events.

These dates cannot be widened through the command line. A new window requires a
new all-ticker provider qualification and a new reviewed package version.

## Exact tickers

`NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA, NFLX, AMD, INTC`

The package refuses missing, additional, or guessed tickers. META uses the
provider symbol recorded in the passed qualification summary.

## Input contracts

### Company reference

Path: `data/reference/company_tickers.csv`

Grain: one curated company and canonical ticker per row.

Required columns: `company`, `ticker`, `aliases`.

### Tiingo qualification summary

Path:
`reports/provider_qualification/tiingo/tiingo_eod_qualification_summary.json`

Required evidence:

- all ten tickers passed;
- requested window equals 2015-01-01 through 2020-04-01;
- internal-use-only licence classification is present;
- every ticker has a passed provider symbol and historical response checksum.

### Sentiment champion

The existing champion and comparison manifests must agree on the deployment
model and label order: Bearish, Neutral, Bullish. Model files are loaded only
from the existing project directory with `local_files_only=True`.

### Secret token

`TIINGO_API_TOKEN` is read only from the environment. It is never placed in a
URL, console message, report, manifest, cache filename, or exception created by
the package.

## Event evidence

Source: official SEC ticker, recent submissions, and historical submissions
APIs.

Grain: one unique canonical ticker and SEC document URL.

Timestamp: official SEC `acceptanceDateTime` in UTC.

Ticker join: canonical ticker to official SEC CIK mapping.

Historical pagination: the main CIK response is parsed first, then every
SEC-declared `filings.files` JSON whose official date range overlaps the fixed
model window is fetched and checksummed. The package does not assume that the
main response contains 2015--2020 filings.

Excluded forms: ownership and holder-reporting forms such as Forms 3, 4, 5,
13F, SC 13D, and SC 13G.

Text: a normalized descriptor built only from SEC company, form, and primary
document description fields. It is not a copied article body.

## Price evidence

Source: Tiingo EOD authenticated API.

Grain: one canonical ticker and trading session.

Model values:

- `open = adjOpen`
- `high = adjHigh`
- `low = adjLow`
- `close = adjClose`
- `volume = adjVolume`

Each response must have:

- at least 1,000 rows;
- all raw and adjusted OHLCV fields;
- dividend cash and split factor;
- unique, strictly increasing dates;
- positive adjusted prices and non-negative volume/dividend values;
- a canonical response checksum matching the passed qualification summary.

Raw response caches are owner-only under `data/private/tiingo_eod` and a nested
`.gitignore` prevents accidental source-control inclusion. Raw Tiingo values
must not be exposed or redistributed by a future public app.

Yahoo Finance through yfinance remains an optional secondary cross-check only.
It is not executed by default and can never replace Tiingo as the sole source in
this package.

## Market-session mapping

Timezone: `America/New_York`.

Regular open: 09:30 local.

Regular close: 16:00 local.

Join rule: map each SEC event to the first actual Tiingo trading session whose
open is strictly later than the SEC acceptance timestamp. Exact-open events map
to the following session. Weekends and holidays come from actual price dates,
not a guessed weekday calendar.

Formula:

`reaction_return = target_adjusted_close / previous_adjusted_close - 1`

Movement labels:

- Down: return < -0.005
- Flat: -0.005 <= return <= 0.005
- Up: return > 0.005

## Leakage protections

- The event timestamp must be earlier than the target-session open.
- Previous close is shifted inside each ticker group only.
- Sentiment inference uses event text and the existing local model.
- Target close, return, and label are outputs, not sentiment inputs.
- Readiness simulates chronological train, validation, and untouched test blocks
  with one purged session at both boundaries.
- Every simulated block must contain Down, Flat, and Up labels.

## Outputs

- `data/processed/news_sentiment_evidence.csv`
- `data/processed/market_price_evidence.csv`
- `data/processed/market_data_foundation_rejected_rows.csv`
- `reports/qa/market_data_foundation_qa.json`
- `artifacts/manifests/market_data_foundation_manifest.json`

All controlled outputs are written atomically with owner-only permissions.
Checksums, sizes, modes, provider requests, qualification evidence, private
cache inventory, readiness, and deployment boundaries are recorded.

## Error handling, caching, and rollback

- SEC recent, SEC historical, and Tiingo requests have short timeouts, visible
  progress, bounded retries,
  Retry-After handling, and safe cache reuse.
- Invalid or incomplete caches fail closed.
- Successful private provider caches remain available for reproducible reruns.
- The strike installer backs up every installed project file and every
  controlled output.
- Any syntax, test, regression, live-build, or verification failure restores
  installed files and controlled outputs automatically.
- Provider caches are not deleted during rollback because they are immutable,
  checksummed, and expensive to reacquire.

## Deployment impact

None. FastAPI, Streamlit, Docker, Kubernetes, CI/CD, and public deployment files
are untouched. The movement model remains disabled until this package prints:

`MARKET DATA FOUNDATION STRIKE PACKAGE: PASSED`
