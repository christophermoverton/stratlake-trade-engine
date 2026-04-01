from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


_TARGET_PATH = Path(__file__).resolve().with_name("milestone_11_5_alpha_portfolio_workflow.py")
_SPEC = spec_from_file_location("milestone_11_5_alpha_portfolio_workflow", _TARGET_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load compatibility module: {_TARGET_PATH}")

_MODULE = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

ExampleArtifacts = _MODULE.ExampleArtifacts
DATASET_PATH = _MODULE.DATASET_PATH
DEFAULT_OUTPUT_ROOT = _MODULE.DEFAULT_OUTPUT_ROOT
FEATURE_COLUMNS = _MODULE.FEATURE_COLUMNS
TARGET_COLUMN = _MODULE.TARGET_COLUMN
EXAMPLE_MODEL_NAME = _MODULE.EXAMPLE_MODEL_NAME
RISK_CONFIG = _MODULE.RISK_CONFIG
ExampleCenteredLinearAlphaModel = _MODULE.ExampleCenteredLinearAlphaModel
load_example_dataset = _MODULE.load_example_dataset
build_fixed_split = _MODULE.build_fixed_split
build_rolling_splits = _MODULE.build_rolling_splits
register_example_model = _MODULE.register_example_model
train_and_predict = _MODULE.train_and_predict
build_cross_section = _MODULE.build_cross_section
run_single_symbol_backtest = _MODULE.run_single_symbol_backtest
build_strategy_return_matrix = _MODULE.build_strategy_return_matrix
build_components = _MODULE.build_components
portfolio_config_dict = _MODULE.portfolio_config_dict
summarize_weights = _MODULE.summarize_weights
write_summary = _MODULE.write_summary
write_csv = _MODULE.write_csv
print_summary = _MODULE.print_summary
run_example = _MODULE.run_example
main = getattr(_MODULE, "main", None)


if __name__ == "__main__":
    if main is not None:
        main()
    else:
        run_example()
