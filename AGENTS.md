# Project constraints
- Follow parent/user governance and competition restrictions.
- Never train, tune, estimate statistics, or select models with test images or leaderboard scores.
- Raw data are immutable and private. Keep them out of Git and remote uploads.
- Preserve official validation split. Record each experiment before starting it.
- Never silently skip corrupt images or invalid rows.
- Run `python tools/verify.py` before delivery; inspect logs and report actual checks.
- User performs final competition upload. Do not push or publish.
