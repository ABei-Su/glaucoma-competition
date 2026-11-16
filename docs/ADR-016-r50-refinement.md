# ADR-016: separate supervised refinement run

Use the existing init_checkpoint API to create a four-epoch fixed low-rate stage from the locked v15 R50. Preserve all v15 artifacts. No training/inference API or persisted trainer schema changes. Explicitly bind supervised ancestry and fresh optimizer, exclude valid from training, prelock epoch4, and compare both shapes to corresponding v15 configs. Independent source freeze and bounded process state evidence; no retry or terminal overwrite. All changes/results exploratory under the user-delegated experiment scope.
