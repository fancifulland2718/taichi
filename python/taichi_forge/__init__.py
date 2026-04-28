from taichi_forge._funcs import *
from taichi_forge._lib import core as _ti_core
from taichi_forge._lib.utils import warn_restricted_version
from taichi_forge._logging import *
from taichi_forge._snode import *
from taichi_forge.lang import *  # pylint: disable=W0622 # TODO(archibate): It's `taichi_forge.lang.core` overriding `taichi_forge.core`
from taichi_forge.types.annotations import *

# Provide a shortcut to types since they're commonly used.
from taichi_forge.types.primitive_types import *


from taichi_forge import ad, algorithms, experimental, graph, linalg, math, sparse, tools, types
from taichi_forge.tools.compile_profile import compile_profile, CompileProfile  # P-Compile-7
from taichi_forge.ui import GUI, hex_to_rgb, rgb_to_hex, ui

# Issue#2223: Do not reorder, or we're busted with partially initialized module
from taichi_forge import aot  # isort:skip

# Explicit submodule re-binding.
#
# `from taichi_forge.lang import *` *does* normally leave the `lang`
# submodule attribute accessible on the parent package, but it can get
# accidentally shadowed by names exported from inner modules (the lang
# subpackage uses a dynamic `__all__ = [s for s in dir() if ...]` which
# is fragile). We also import these here so that `taichi_forge.lang`,
# `taichi_forge.lang.impl`, etc. are always reachable as documented —
# downstream code commonly does `ti.lang.impl.current_cfg()`.
from taichi_forge import lang  # noqa: E402,F401  pylint: disable=C0411,C0413


def __getattr__(attr):
    if attr == "cfg":
        return None if lang.impl.get_runtime().prog is None else lang.impl.current_cfg()
    raise AttributeError(f"module '{__name__}' has no attribute '{attr}'")


__version__ = (
    _ti_core.get_version_major(),
    _ti_core.get_version_minor(),
    _ti_core.get_version_patch(),
)

del _ti_core

warn_restricted_version()
del warn_restricted_version
