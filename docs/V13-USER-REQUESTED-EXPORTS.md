# User-requested v13 trial exports, 2026-10-29
The user explicitly requests two NEW predictions even if quality is worse.
Export the final chronological fold (fold2), epoch8: control and mixup separately,
using single-model horizontal TTA. No model is selected with test or leaderboard.
Both models trained on 34264 official train images; neither is a full-train model.
Original v13 remains not_promoted. This independent trial export does not override
its gate or claim promotion. Preserve all frozen sources and all historical CSVs.
Use existing predict.py, UTF8 CSV with official order, 5855 rows and threshold .5.
Record config/model hashes before test forward. Official T4 timing unverified.
