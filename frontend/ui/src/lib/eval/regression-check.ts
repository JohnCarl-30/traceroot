/**
 * A CI regression check over one candidate-vs-baseline comparison (issue #2763).
 *
 * Pure: it takes `compareRuns` output plus a caller's policy and returns a
 * machine-readable pass/fail decision. It adds no comparison arithmetic of its own —
 * means, directions and cell classifications all come from `compareRuns`.
 *
 * It fails closed: anything that keeps the comparison from being authoritative
 * (a non-trustworthy run pair, a missing or errored score, an unknown metric) is a
 * failure, never a silent pass.
 */

import type {
  CompareRunsOutput,
  ComparisonState,
  RunComparabilityReason,
  ScorerDirection,
} from "./comparison";

export interface RegressionPolicy {
  /** The emitted metric (scorer) name to check. */
  metric: string;
  /** Largest allowed worsening of the metric's mean, in the metric's own units (>= 0). */
  maxRegression: number;
}

export type RegressionCheckFailure =
  | "comparison_not_trustworthy"
  | "metric_not_found"
  | "metric_not_directional"
  | "incomplete_scores"
  | "no_paired_scores"
  | "regression_exceeds_tolerance";

export interface RegressionCheckResult {
  passed: boolean;
  failure: RegressionCheckFailure | null;
  metric: string;
  direction: ScorerDirection | null;
  candidateMean: number | null;
  baselineMean: number | null;
  /** How much worse the candidate is by the metric's direction: positive = worse, negative = better. */
  regression: number | null;
  maxRegression: number;
  pairedCount: number;
  /** This metric's cells that could not be compared (missing, errored, version change, unpaired). */
  notComparableCount: number;
  state: ComparisonState;
  reasons: RunComparabilityReason[];
}

// Float noise: 0.90 - 0.87 is 0.030000000000000027, which must not breach a 0.03 tolerance.
const EPSILON = 1e-9;

export function checkRegression(
  output: CompareRunsOutput,
  policy: RegressionPolicy,
): RegressionCheckResult {
  if (!Number.isFinite(policy.maxRegression) || policy.maxRegression < 0) {
    throw new RangeError("maxRegression must be a finite number >= 0");
  }

  const { comparison, results } = output;
  const aggregate = comparison.scorers.find((s) => s.name === policy.metric);

  let notComparableCount = 0;
  for (const result of results) {
    for (const cell of result.scorerCells) {
      if (
        cell.scorerName === policy.metric &&
        (cell.classification === "not_comparable" || cell.classification === "unpaired")
      ) {
        notComparableCount += 1;
      }
    }
  }

  const base: Omit<RegressionCheckResult, "passed" | "failure"> = {
    metric: policy.metric,
    direction: aggregate?.direction ?? null,
    candidateMean: aggregate?.candidateMean ?? null,
    baselineMean: aggregate?.baselineMean ?? null,
    regression: null,
    maxRegression: policy.maxRegression,
    pairedCount: aggregate?.pairedCount ?? 0,
    notComparableCount,
    state: comparison.state,
    reasons: [...comparison.reasons],
  };
  const fail = (failure: RegressionCheckFailure): RegressionCheckResult => ({
    ...base,
    passed: false,
    failure,
  });

  if (comparison.state !== "trustworthy") return fail("comparison_not_trustworthy");
  if (!aggregate) return fail("metric_not_found");
  if (aggregate.direction === "none" || aggregate.valueType === "categorical") {
    return fail("metric_not_directional");
  }
  if (notComparableCount > 0) return fail("incomplete_scores");
  if (aggregate.candidateMean === null || aggregate.baselineMean === null) {
    return fail("no_paired_scores");
  }

  const regression =
    aggregate.direction === "higher_is_better"
      ? aggregate.baselineMean - aggregate.candidateMean
      : aggregate.candidateMean - aggregate.baselineMean;
  const checked = { ...base, regression };

  if (regression > policy.maxRegression + EPSILON) {
    return { ...checked, passed: false, failure: "regression_exceeds_tolerance" };
  }
  return { ...checked, passed: true, failure: null };
}
