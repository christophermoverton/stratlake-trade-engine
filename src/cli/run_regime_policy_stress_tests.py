from __future__ import annotations

import argparse
import sys
from typing import Sequence

from src.config.regime_policy_stress_tests import RegimePolicyStressTestConfigError


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic adaptive regime-policy stress tests."
    )
    parser.add_argument("--config", required=True, help="Regime policy stress-test YAML/JSON config path.")
    parser.add_argument("--source-review-pack", help="Optional source review-pack directory override.")
    parser.add_argument("--policy-metrics-path", help="Optional policy metrics CSV/JSON override.")
    parser.add_argument("--output-root", help="Optional stress-test output root override.")
    return parser.parse_args(argv)


def run_cli(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    from src.execution.regime_policy_stress_tests import run_regime_policy_stress_tests_from_cli_args

    result = run_regime_policy_stress_tests_from_cli_args(args).raw_result
    print_summary(result)
    return result


def print_summary(result) -> None:
    print("Regime Policy Stress Test Summary")
    print("---------------------------------")
    print(f"Stress run id: {result.stress_run_id}")
    print(f"Output directory: {result.output_dir.as_posix()}")
    print(f"Scenario count: {result.scenario_summary.get('scenario_count', 0)}")
    print(f"Policy count: {result.scenario_summary.get('policy_count', 0)}")
    print(f"Most resilient policy: {result.policy_stress_summary.get('most_resilient_policy')}")
    print(f"stress_matrix.csv: {result.stress_matrix_csv_path.as_posix()}")
    print(f"policy_stress_summary.json: {result.policy_stress_summary_path.as_posix()}")


def main() -> None:
    try:
        run_cli()
    except (RegimePolicyStressTestConfigError, ValueError, FileNotFoundError) as exc:
        print(str(exc).strip() or exc.__class__.__name__, file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
