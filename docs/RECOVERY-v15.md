# v15 single authorized startup recovery

Original trainer54916 and watcher46564 disappeared during content audit; host CIM confirmed neither nor any Python process existed, both output/error logs remained empty, no run directory/checkpoint existed. Cause of external process loss is unknown; do not claim a model failure or completed training. The user explicitly approved one recovery, same recipe, original20:37:47 deadline, no lock clearing or budget increase.

Evidence copied under output/v15/recovery_001/record.json. New tools/recover_v15_once.ps1 checks original PIDs absent, original run absent, review/source SHA, original43 input SHA, positive remaining budget and exclusive recovery reservation. It uses the unchanged train.py arguments and records v15_recovery_launch.json, v15_train_recovery001.log/stderr and normal v15_exit.json required by fixed finalizer. No further automatic recovery. An independently created hidden host process will own the bounded wrapper; it does not install a recurring system task or service.

Remaining budget is computed from the original deadline. Success invokes the already frozen v15 finalizer watchdog. Completion still requires independent delivery review. Source/selection/seed/epoch/TTA/threshold remain unchanged.
