"""Evidence localization and cropping (Phase 5)."""

from .localize import Box, Region, heatmap, evidence_regions, max_patch_region, to_pixels
from .crop import render_region, crop_image

__all__ = ['Box', 'Region', 'heatmap', 'evidence_regions', 'max_patch_region', 'to_pixels', 'render_region', 'crop_image']
