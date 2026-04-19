from __future__ import annotations

from pathlib import Path

from src.pipeline.builder import PipelineBuilder


def main() -> None:
    output_path = Path("configs/generated/pipeline_builder_example.yml")
    builder = (
        PipelineBuilder("pipeline_builder_example")
        .strategy("cross_section_momentum", params={"lookback_days": 20})
        .signal("cross_section_rank")
        .construct_positions(
            "rank_dollar_neutral",
            params={"gross_long": 0.5, "gross_short": 0.5},
        )
        .portfolio("equal_weight", params={"timeframe": "1D"})
    )
    builder.to_yaml(output_path)
    print(output_path.as_posix())


if __name__ == "__main__":
    main()
