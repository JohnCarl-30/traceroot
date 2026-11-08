import { describe, it, expect } from "vitest";
import {
  compareRuns,
  type ComparisonRun,
  type ComparisonResult,
  type ComparisonScore,
  type ComparisonScorerMeta,
} from "./comparison";
import { checkRegression } from "./regression-check";

// ── builders ─────────────────────────────────────────────────────────────

function score(name: string, opts: Partial<ComparisonScore> = {}): ComparisonScore {
  return {
    scorerName: name,
    scorerVersion: opts.scorerVersion ?? "unversioned",
    numericValue: opts.numericValue ?? null,
    boolValue: opts.boolValue ?? null,
    stringValue: opts.stringValue ?? null,
    error: opts.error ?? null,
  };
}
const num = (name: string, v: number) => score(name, { numericValue: v });

function result(testCaseId: string, scores: ComparisonScore[]): ComparisonResult {
  return { testCaseId, status: "passed", candidateOutput: null, durationMs: null, scores };
}

function run(opts: Partial<ComparisonRun> = {}): ComparisonRun {
  return {
    id: opts.id ?? "run_cand",
    runNumber: opts.runNumber ?? 2,
    evaluationId: opts.evaluationId ?? "eval_1",
    datasetVersionId: opts.datasetVersionId ?? "dsv_1",
    candidateVersion: opts.candidateVersion ?? "candidate",
    status: opts.status ?? "completed",
    baselineRunId: opts.baselineRunId === undefined ? "run_base" : opts.baselineRunId,
    scorers: opts.scorers ?? [{ name: "acc", version: "unversioned" }],
  };
}

/** One numeric score per case on each side; case ids are "c0", "c1", ... */
function compareNumeric(
  candidate: number[],
  baseline: number[],
  opts: { scorers?: ComparisonScorerMeta[]; candidateRun?: Partial<ComparisonRun> } = {},
) {
  const metric = opts.scorers?.[0]?.name ?? "acc";
  return compareRuns({
    candidate: run({ scorers: opts.scorers, ...opts.candidateRun }),
    candidateResults: candidate.map((v, i) => result(`c${i}`, [num(metric, v)])),
    baseline: run({ id: "run_base", runNumber: 1, scorers: opts.scorers }),
    baselineResults: baseline.map((v, i) => result(`c${i}`, [num(metric, v)])),
  });
}

// ── tolerance ────────────────────────────────────────────────────────────

describe("checkRegression tolerance", () => {
  it("fails when a higher-is-better mean drops by more than the tolerance (0.90 → 0.87, tolerance 0.02)", () => {
    const check = checkRegression(compareNumeric([0.87], [0.9]), {
      metric: "acc",
      maxRegression: 0.02,
    });
    expect(check.passed).toBe(false);
    expect(check.failure).toBe("regression_exceeds_tolerance");
    expect(check.regression).toBeCloseTo(0.03);
  });

  it("passes a drop exactly at the tolerance despite float noise (0.90 → 0.87, tolerance 0.03)", () => {
    const check = checkRegression(compareNumeric([0.87], [0.9]), {
      metric: "acc",
      maxRegression: 0.03,
    });
    expect(check.passed).toBe(true);
    expect(check.failure).toBeNull();
  });

  it("reports an improvement as a negative regression and passes a zero tolerance", () => {
    const check = checkRegression(compareNumeric([0.95, 0.85], [0.8, 0.8]), {
      metric: "acc",
      maxRegression: 0,
    });
    expect(check.passed).toBe(true);
    expect(check.regression).toBeCloseTo(-0.1);
    expect(check.candidateMean).toBeCloseTo(0.9);
    expect(check.baselineMean).toBeCloseTo(0.8);
    expect(check.pairedCount).toBe(2);
  });

  it("flips the direction for lower-is-better metrics", () => {
    const scorers = [
      { name: "latency", version: "unversioned", direction: "lower_is_better" as const },
    ];
    const slower = checkRegression(compareNumeric([130], [100], { scorers }), {
      metric: "latency",
      maxRegression: 20,
    });
    expect(slower.failure).toBe("regression_exceeds_tolerance");
    expect(slower.regression).toBe(30);

    const faster = checkRegression(compareNumeric([90], [100], { scorers }), {
      metric: "latency",
      maxRegression: 0,
    });
    expect(faster.passed).toBe(true);
    expect(faster.regression).toBe(-10);
  });

  it("checks a boolean metric as a pass rate", () => {
    const out = compareRuns({
      candidate: run({ scorers: [{ name: "ok", version: "unversioned" }] }),
      candidateResults: [
        result("a", [score("ok", { boolValue: true })]),
        result("b", [score("ok", { boolValue: false })]),
      ],
      baseline: run({
        id: "run_base",
        runNumber: 1,
        scorers: [{ name: "ok", version: "unversioned" }],
      }),
      baselineResults: [
        result("a", [score("ok", { boolValue: true })]),
        result("b", [score("ok", { boolValue: true })]),
      ],
    });
    const check = checkRegression(out, { metric: "ok", maxRegression: 0.25 });
    expect(check.failure).toBe("regression_exceeds_tolerance");
    expect(check.regression).toBeCloseTo(0.5);
  });
});

// ── fail closed ──────────────────────────────────────────────────────────

describe("checkRegression fails closed", () => {
  it("fails when there is no baseline", () => {
    const out = compareRuns({
      candidate: run({ baselineRunId: null }),
      candidateResults: [result("c0", [num("acc", 0.9)])],
      baseline: null,
      baselineResults: [],
    });
    const check = checkRegression(out, { metric: "acc", maxRegression: 1 });
    expect(check.failure).toBe("comparison_not_trustworthy");
    expect(check.state).toBe("unavailable");
    expect(check.reasons).toEqual(["no_baseline"]);
  });

  it("fails while the candidate is still running", () => {
    const check = checkRegression(
      compareNumeric([0.9], [0.9], { candidateRun: { status: "running" } }),
      { metric: "acc", maxRegression: 1 },
    );
    expect(check.failure).toBe("comparison_not_trustworthy");
    expect(check.state).toBe("pending");
  });

  it("fails when the runs used different dataset versions", () => {
    const check = checkRegression(
      compareNumeric([0.9], [0.9], { candidateRun: { datasetVersionId: "dsv_2" } }),
      { metric: "acc", maxRegression: 1 },
    );
    expect(check.failure).toBe("comparison_not_trustworthy");
    expect(check.reasons).toContain("different_dataset_version");
  });

  it("fails when the runs scored different case sets", () => {
    const out = compareRuns({
      candidate: run(),
      candidateResults: [result("c0", [num("acc", 0.9)])],
      baseline: run({ id: "run_base", runNumber: 1 }),
      baselineResults: [result("c0", [num("acc", 0.9)]), result("c1", [num("acc", 0.9)])],
    });
    const check = checkRegression(out, { metric: "acc", maxRegression: 1 });
    expect(check.failure).toBe("comparison_not_trustworthy");
    expect(check.reasons).toContain("case_set_mismatch");
  });

  it("fails when a case's score errored, instead of averaging over the rest", () => {
    const out = compareRuns({
      candidate: run(),
      candidateResults: [
        result("c0", [num("acc", 0.9)]),
        result("c1", [score("acc", { error: "judge timed out" })]),
      ],
      baseline: run({ id: "run_base", runNumber: 1 }),
      baselineResults: [result("c0", [num("acc", 0.9)]), result("c1", [num("acc", 0.9)])],
    });
    const check = checkRegression(out, { metric: "acc", maxRegression: 1 });
    expect(check.failure).toBe("incomplete_scores");
    expect(check.notComparableCount).toBe(1);
  });

  it("fails for a metric neither run reported", () => {
    const check = checkRegression(compareNumeric([0.9], [0.9]), {
      metric: "faithfulness",
      maxRegression: 1,
    });
    expect(check.failure).toBe("metric_not_found");
  });

  it("fails for a categorical metric, which has no direction to regress in", () => {
    const out = compareRuns({
      candidate: run({ scorers: [{ name: "label", version: "unversioned" }] }),
      candidateResults: [result("c0", [score("label", { stringValue: "billing" })])],
      baseline: run({
        id: "run_base",
        runNumber: 1,
        scorers: [{ name: "label", version: "unversioned" }],
      }),
      baselineResults: [result("c0", [score("label", { stringValue: "refund" })])],
    });
    const check = checkRegression(out, { metric: "label", maxRegression: 1 });
    expect(check.failure).toBe("metric_not_directional");
  });

  it("fails for a declared metric with no paired scores", () => {
    const out = compareRuns({
      candidate: run({ scorers: [{ name: "acc", version: "unversioned" }] }),
      candidateResults: [result("c0", [])],
      baseline: run({ id: "run_base", runNumber: 1 }),
      baselineResults: [result("c0", [])],
    });
    const check = checkRegression(out, { metric: "acc", maxRegression: 1 });
    expect(check.failure).toBe("no_paired_scores");
    expect(check.pairedCount).toBe(0);
  });

  it("rejects a negative or non-finite tolerance", () => {
    const out = compareNumeric([0.9], [0.9]);
    expect(() => checkRegression(out, { metric: "acc", maxRegression: -0.01 })).toThrow(RangeError);
    expect(() => checkRegression(out, { metric: "acc", maxRegression: Number.NaN })).toThrow(
      RangeError,
    );
  });
});
