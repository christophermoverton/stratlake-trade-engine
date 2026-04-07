from __future__ import annotations

# Dedicated module path for the cross-sectional LightGBM alpha case study.
# The concrete implementation is re-exported here so the case study has a
# stable, discoverable implementation file without changing the public
# built-in alpha interfaces that the rest of the research stack already imports.

from src.research.alpha.builtins import CrossSectionalLightGBMAlphaModel, LightGBMModelSpec

__all__ = ["CrossSectionalLightGBMAlphaModel", "LightGBMModelSpec"]
