# ADR-017: isolated one-factor BCE weighting experiment

Change only positive class training weight from1 to2 relative to v15's fresh8-epoch R50. Use existing trainer API and math; no inference or schema changes, no threshold calibration. Freeze before new outcomes and compare to direct v15 configurations. Keep source/data/history/previous submissions immutable. Independent bounded new paths and explicit review before delivery. Probability shifts alone are not the quality criterion. Exploration on reused validation is disclosed.
