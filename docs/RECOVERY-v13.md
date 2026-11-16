# v13 recovery 001

2026-10-29: user explicitly requested continuation after interruption disclosure.
Host CIM repeatedly found coordinator9544 and trainer4084 absent; the host did not
reboot. Logs ended during fold2 control epoch4 without a caught traceback; exact
termination cause remains unknown. No claim of a diagnosed CUDA fault is made.

CPU audit verified all180 frozen evidence items, four completed fits, epoch3
checkpoint config/binding, finite weights, optimizer/scheduler/scaler and all RNG
entries. Original checkpoint SHA256:
04fb5c6c0a6f67a2bff0422df66566bf7ad42758fc07e21a0e1b2ef2f033bdb8.

The separate one-shot recovery coordinator archives exact old state/locks/log
metadata and checkpoint, then acquires a new owner and invokes the unchanged
trainer with --resume. It preserves the original12hour campaign deadline and
the original fold2control3hour deadline, including downtime. No resource rerun,
recipe change, earlier-fit replay or source-freeze update occurs. It then runs
the remaining declared fold2Mixup and the same gate/full/finalization path.

Recovery is explicit, user-authorized and single use; a second interruption is
not automatically retried. CUDA/four-worker bitwise replay equivalence was not
experimentally proven; the implementation restores stored RNG/optimizer states
and repeats only the incomplete epoch. All old evidence and submissions remain.
