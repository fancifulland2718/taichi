"""Cold, cuFFT-owned store-scaling compiler; not a general Forge JIT backend.

Only explicit callback preparation imports this module. The caller supplies
the optional runtime libraries; replay uses the ordinary retained cuFFT plan.
"""

import ctypes
import hashlib
import json
import os
from pathlib import Path

from taichi_forge.hardware._fft import _CufftPlanBase
from taichi_forge.lang.exception import TaichiRuntimeError


def _library(path, label):
    if path is None:
        raise ValueError(
            f"cuFFT LTO preparation requires an explicit {label} library path"
        )
    resolved = Path(path).expanduser().resolve(strict=True)
    library = ctypes.CDLL(str(resolved))
    if os.name == "nt":
        # Report the binary that Windows actually mapped, not merely the
        # requested path (a same-name module may already be resident).
        filename = ctypes.WinDLL("kernel32", use_last_error=True).GetModuleFileNameW
        filename.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint)
        filename.restype = ctypes.c_uint
        buffer = ctypes.create_unicode_buffer(32768)
        if not filename(library._handle, buffer, len(buffer)):
            raise ctypes.WinError(ctypes.get_last_error())
        resolved = Path(buffer.value).resolve(strict=True)
    return library, {
        "path": str(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


class _StoreScaleCallback:
    symbol = "forge_cufft_store_scale"

    def __init__(self, scale, *, nvrtc_library, nvjitlink_library, expected=None):
        nvrtc, compiler = _library(nvrtc_library, "NVRTC")
        linker, link_facts = _library(nvjitlink_library, "nvJitLink")
        # Keep explicit libraries live through plan creation; no global loader,
        # environment modification, background activity or replay dependency.
        self._libraries = (nvrtc, linker)
        version = nvrtc.nvrtcVersion
        version.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
        major, minor = ctypes.c_int(), ctypes.c_int()
        self._check(version(ctypes.byref(major), ctypes.byref(minor)), "version query")
        compiler["version"] = [major.value, minor.value]
        link_version = linker.nvJitLinkVersion
        link_version.argtypes = (
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint),
        )
        link_major, link_minor = ctypes.c_uint(), ctypes.c_uint()
        if link_version(ctypes.byref(link_major), ctypes.byref(link_minor)):
            raise TaichiRuntimeError("cuFFT callback nvJitLink version query failed")
        link_facts["version"] = [link_major.value, link_minor.value]
        if major.value != link_major.value or minor.value > link_minor.value:
            raise ValueError(
                "cuFFT callbacks require same-major NVRTC no newer than nvJitLink"
            )
        options = (
            "--std=c++17",
            "--gpu-architecture=compute_75",
            "--device-c",
            "-dlto",
        )
        # NVRTC's builtin float2 gives the cuFFT complex-f32 ABI without Toolkit
        # headers. No callerInfo allocation: scale is an immutable f32 constant.
        source = f"""
__device__ void {self.symbol}(void* output, unsigned long long offset,
                             float2 value, void*, void*) {{
    const float scale = {float(scale).hex()}f;
    value.x *= scale;
    value.y *= scale;
    static_cast<float2*>(output)[offset] = value;
}}
"""
        facts = {
            "schema": "cufft-store-scale-lto-v1",
            "symbol": self.symbol,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "compiler": compiler,
            "linker": link_facts,
            "options": options,
            "caller_info_bytes": 0,
        }
        if expected is not None:
            prior = {
                key: value for key, value in expected.items() if key != "ir_sha256"
            }
            if json.dumps(facts, sort_keys=True) != json.dumps(prior, sort_keys=True):
                raise TaichiRuntimeError(
                    "FFT callback compiler, linker or source contract drifted"
                )
        program = ctypes.c_void_p()
        create = nvrtc.nvrtcCreateProgram
        create.argtypes = (
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        compile_program = nvrtc.nvrtcCompileProgram
        compile_program.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        )
        destroy = nvrtc.nvrtcDestroyProgram
        destroy.argtypes = (ctypes.POINTER(ctypes.c_void_p),)
        self._check(
            create(
                ctypes.byref(program),
                source.encode(),
                b"forge_fft_store.cu",
                0,
                None,
                None,
            ),
            "program creation",
        )
        try:
            encoded = (ctypes.c_char_p * len(options))(
                *(value.encode() for value in options)
            )
            status = compile_program(program, len(options), encoded)
            if status:
                log = (
                    self._read(nvrtc, program, "ProgramLog")
                    .rstrip(b"\0")
                    .decode(errors="replace")
                )
                raise TaichiRuntimeError(
                    f"cuFFT store callback compilation failed ({status}): {log}"
                )
            self.ir = self._read(nvrtc, program, "LTOIR")
        finally:
            destroy(ctypes.byref(program))
        facts["ir_sha256"] = hashlib.sha256(self.ir).hexdigest()
        if expected is not None and facts["ir_sha256"] != expected.get("ir_sha256"):
            raise TaichiRuntimeError("Recompiled FFT callback LTO identity drifted")
        self._facts_json = json.dumps(facts, sort_keys=True)

    @staticmethod
    def _check(status, phase):
        if status:
            raise TaichiRuntimeError(f"cuFFT callback NVRTC {phase} failed ({status})")

    @classmethod
    def _read(cls, library, program, kind):
        get_size = getattr(library, f"nvrtcGet{kind}Size")
        get_size.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t))
        size = ctypes.c_size_t()
        cls._check(get_size(program, ctypes.byref(size)), f"{kind} size")
        data = ctypes.create_string_buffer(size.value)
        get_data = getattr(library, f"nvrtcGet{kind}")
        get_data.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        cls._check(get_data(program, data), kind)
        return data.raw

    @property
    def facts(self):
        return json.loads(self._facts_json)


class _CufftStoreScalePlan(_CufftPlanBase):

    @classmethod
    def prepare(
        cls,
        dimensions,
        batch_count,
        scale,
        *,
        nvrtc_library=None,
        nvjitlink_library=None,
        expected=None,
    ):
        """Keep toolchain reconstruction out of the mathematical FFT catalog."""
        if expected is not None:
            nvrtc_library = expected["compiler"]["path"]
            nvjitlink_library = expected["linker"]["path"]
        return cls(
            dimensions,
            batch_count,
            _StoreScaleCallback(
                scale,
                nvrtc_library=nvrtc_library,
                nvjitlink_library=nvjitlink_library,
                expected=expected,
            ),
        )

    @property
    def callback_facts(self):
        return self._callback.facts

    def __init__(self, dimensions, batch_count, callback):
        # Ownership remains with the same completion-retained native plan.
        self._callback = callback
        self._initialize(
            dimensions,
            batch_count=batch_count,
            transform="c2c",
            _store_callback=callback,
        )

    def _graph_provider_memory_identity(self):
        return (
            *super()._graph_provider_memory_identity(),
            "store_scale_lto",
            self._callback.facts["ir_sha256"],
        )
