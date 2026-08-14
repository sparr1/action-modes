"""TD-MPC2's custom single-task DMControl domains.

The task definitions and XML assets in this package are vendored from
nicklashansen/tdmpc2 at commit 8bbc14ebabdb32ea7ada5c801dc525d0dc73bafe.
Importing this package registers the tasks in the corresponding DMControl
domain suites.  The top-level :mod:`domains` package deliberately does not
import this package so DMControl remains an optional dependency.
"""

from . import ball_in_cup, cheetah, fish, hopper, pendulum, reacher, walker

__all__ = (
    "ball_in_cup",
    "cheetah",
    "fish",
    "hopper",
    "pendulum",
    "reacher",
    "walker",
)
