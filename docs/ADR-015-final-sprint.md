# ADR-015: independent fresh R50 final trial

Use unchanged existing training and prediction APIs, fixed epoch8, new v15 paths, and two predeclared outputs. Old v14 and all historical evidence remain intact. This tests a fresh single-stage recipe as a whole; seed, learning rate and initialization differences prevent attributing any change solely to one factor. No leaderboard fitting or official-valid training. Outer guards own only reservations/timeouts/exit evidence; trainer owns training lifecycle; finalizer owns its independent state machine. No automatic retry or terminal overwrite.
