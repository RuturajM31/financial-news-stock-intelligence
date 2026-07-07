# Movement Intelligence Stability Contract v8.3

## Purpose and order

The package performs these stages in one rollback-safe transaction:

1. verify the exact local Market Data Foundation v8;
2. verify one compatible OpenMP runtime family for movement training;
3. run focused tests and isolated full-project regression;
4. build one-row-per-ticker-session leakage-safe model evidence;
5. rank candidates on four purged expanding development folds;
6. freeze a diverse shortlist before terminal labels are scored;
7. fit one OOF-only global policy per shortlisted candidate;
8. run the shortlist on a known terminal development block;
9. select the strongest candidate that passes unchanged terminal gates;
10. evaluate the known historical audit once after all development decisions;
11. build explainability and intelligence only after all movement gates pass;
12. run isolated full-project regression after successful installation.

## Input grain, joins, and provenance

- SEC input grain: one verified disclosure mapped to one target session.
- Price input grain: one canonical ticker and adjusted daily session.
- Model grain: one canonical ticker and target-session date.
- Event aggregation key: `ticker + target_session_date`.
- Price-feature join key: `ticker + target_session_date`.
- Joins are validated as one-to-one at model grain.
- Raw provider caches, tokens, source responses, and restricted price values are
  never copied into diagnostics.

## Preprocessing and leakage boundaries

- SEC text must be published before the target market open.
- TF-IDF vocabulary, imputers, encoders, and scalers fit inside the relevant
  training fold or approved refit block only.
- Market features are shifted by at least one complete session.
- Event-history features exclude the current event reaction.
- Target return, target prices, target volume, labels, future timestamps, and
  raw source URLs are forbidden model inputs.
- Dates are chronologically ordered and never randomly shuffled.

## Development selection

The original train and validation blocks form development evidence. The final
9% of unique development dates, with a minimum of 60 dates, form a known
terminal development block. At least one complete date is purged before it.

Candidate selection starts with four purged expanding folds in the earlier
selection block. Identity decisions are used for candidate ranking so one
fold's labels cannot tune the policy used to score that same fold.

Candidates first pass the unchanged weakest-fold and pooled development gates.
Eligible candidates are ranked using fixed chronological weights `[1, 2, 3, 4]`
so later development periods have more influence without receiving access to
terminal or historical-audit labels.

A shortlist contains at most five learned candidates and at most two candidates
from one family. The shortlist is frozen before terminal labels are scored.

## Candidate-specific OOF policy calibration

Each shortlisted candidate receives a policy fitted only from that candidate's
selection-fold OOF probabilities and labels. The terminal block and historical
audit are not accepted as policy inputs.

The Down offset is fixed to zero. Flat and Up offsets use the fixed grid
`[-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]`.

A non-identity policy qualifies only when:

- pooled macro-F1 improvement is at least `0.01`;
- pooled weighted F1 does not decline;
- all three classes are predicted in every fold;
- at least half of folds improve weighted F1;
- no fold's weighted F1 falls by more than `0.015` from identity.

Ticker-specific offsets remain disabled. Identity is the deterministic fallback.

## Terminal development tournament

The terminal block is known from prior runs and is explicitly marked
non-pristine. It is used as additional development evidence, not described as
an independent future-performance estimate.

Only candidates frozen in the earlier shortlist may enter. Every candidate is
fitted on the earlier selection block, scored with its frozen OOF policy, and
checked against the unchanged terminal macro-F1, baseline-improvement, and
three-class gates. Passing candidates are ranked by terminal weighted F1,
terminal macro F1, and previously computed rolling evidence.

The historical audit is not available to shortlist construction, policy
calibration, terminal ranking, candidate definitions, thresholds, or features.

## Candidate and convergence contract

Candidate families include prior baseline, LBFGS logistic regression, recency
and ticker-weighted variants, random forest, ExtraTrees, RBF SVC, SGD log-loss,
and fixed soft-voting ensembles.

V8 adds recency-weighted random-forest and ExtraTrees variants with 365-day and
730-day half-lives. Weights use only dates inside the current training fold and
are normalized to a mean of one.

V8 also adds fixed RF/SGD, RF/ExtraTrees, and RF/ExtraTrees/SGD soft votes. Their
components and weights are declared in source before terminal or audit scoring.

- Calibrated LinearSVC remains excluded.
- `ConvergenceWarning` is a candidate or refit failure.
- Other warning categories remain visible.
- Successful iterative estimators record configured and observed iterations.
- Failed candidates remain in diagnostics and cannot be selected.
- Logistic regression uses LBFGS, `max_iter=5000`, and `tol=1e-4`.

## OpenMP and regression isolation

Movement preflight records native libraries after NumPy, pandas, scikit-learn,
movement-dataset, and movement-training imports.

- zero or one OpenMP family is accepted;
- multiple runtime families fail before fitting;
- thread-count limits do not waive a conflict;
- `KMP_DUPLICATE_LIB_OK` is forbidden.

Regression preserves complete test-file coverage while separating native
library domains. BERT, DistilBERT, and LoRA files use the dedicated Transformer
environment. Ambiguous Transformer data tests are collection-probed. All other
tests use the analytics environment. Every file runs once in a fresh process.

## Unchanged acceptance gates

- development macro F1: at least `0.34`;
- development macro-F1 improvement over baseline: at least `0.015`;
- minimum fold macro F1: at least `0.25`;
- minimum fold weighted F1: at least `0.30`;
- three predicted classes in every required development evaluation;
- historical-audit macro F1: at least `0.30`;
- historical-audit weighted F1: at least `0.40`;
- each actual class has at least five rows and receives predictions;
- each ticker with at least eight rows has macro F1 of at least `0.10`.

No code path lowers a gate after failure.

## Reference-only sentiment phrase coverage

Sentiment phrase extraction uses train and validation SEC events only. When all
three hard sentiment labels occur in that reference period, phrases use the
original class-mean TF-IDF contrast.

If one hard sentiment label is absent, the package does not read test rows,
copy audit labels, or invent a missing class. It uses the verified
`prob_bearish`, `prob_neutral`, and `prob_bullish` values already stored on the
reference events as soft weights. For sentiment class `c`, the primary score is
the weighted TF-IDF mean for `c` minus the weighted mean for `1 - c`. If fewer
than three positive contrasts exist, the remaining rows use the highest
positive class-weighted relevance and are explicitly marked with
`probability_weighted_relevance`.

The fallback fails closed when probabilities are missing, non-finite, outside
`[0, 1]`, do not sum to one, or provide no class-versus-complement contrast.
Outputs record hard-label counts, effective probability weight, selection
basis, method, and whether fallback was used.

## Outputs and downstream use

Movement outputs include the serialized champion, model table, historical-audit
predictions, movement metrics, and external diagnostics. Intelligence consumes
them only after independent movement verification succeeds.

Diagnostics include candidate aggregates, every rolling fold, convergence
evidence, policy evidence, terminal tournament evidence, historical-audit
metrics and predictions, native-runtime trace, summary, checksums, sizes, and
permissions.

## Failure, cleanup, and rollback

- controlled files use atomic same-directory replacement;
- temporary files and transaction locks are removed;
- every controlled source file and output is backed up before installation;
- any error or interruption restores the pre-strike project state;
- external diagnostic evidence intentionally survives rollback;
- deployment and application infrastructure remain unchanged.

The terminal block and historical audit are known rather than pristine. A pass
means the unchanged project contract was satisfied; it does not guarantee
future market performance.
