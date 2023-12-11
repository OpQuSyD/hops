"""
The signal_delay module provides a decorator to protect a function
from being interrupted via signaling mechanism.

When a protected function receives a signal, the signal will be stored
and emitted again AFTER the function has terminated.
"""

import logging
import traceback
import signal
import os

log = logging.getLogger(__name__)

SIG_MAP = {}
for s in signal.__dict__:
    if s.startswith("SIG") and s[3] != "_":
        SIG_MAP[getattr(signal, s)] = s


class SigHandler(object):
    """a handler class which
    - stores the receives signals
    - and allows to re-emit them via os.kill
    """

    def __init__(self):
        self.sigs_caught = []

    def __call__(self, sig, frame):  # a callable suitable to signal.signal
        log.info("caught sig '{}'".format(SIG_MAP[sig]))
        log.debug("frame: {}".format(traceback.format_stack(frame)))
        self.sigs_caught.append(sig)

    def emit(self):  # emit the signals
        l = len(self.sigs_caught)
        if l > 0:
            log.info("caught {} signal(s)".format(l))
            for s in self.sigs_caught:
                log.info("emit signal '{}'".format(SIG_MAP[s]))
                os.kill(os.getpid(), s)


class sig_delay(object):
    """the decorator object, init takes a list of signals which will be delayed"""

    def __init__(self, sigs, handler=None):
        self.sigs = sigs
        self.handler = handler

    def _setup(self):
        self.sigh = SigHandler()
        self.old_handlers = []
        log.debug(
            "setup alternative signal handles for {}".format(
                [SIG_MAP[s] for s in self.sigs]
            )
        )
        for s in self.sigs:
            self.old_handlers.append(signal.getsignal(s))
            signal.signal(s, self.sigh)

    def _restore(self):
        log.debug("restore signal handles")
        for i, s in enumerate(self.sigs):
            signal.signal(s, self.old_handlers[i])
        self.sigh.emit()

    def __call__(self, func):
        def _protected_func(*args, **kwargs):
            with self:
                log.debug("call function ...")
                func(*args, **kwargs)

        return _protected_func

    def __enter__(self):
        self._setup()
        log.debug("signal_delay context entered ...")

    def __exit__(self, exc_type, exc_val, exc_tb):
        log.debug("signal_delay context left!")
        if len(self.sigh.sigs_caught) > 0 and self.handler is not None:
            self.handler(self.sigh.sigs_caught)

        self._restore()
