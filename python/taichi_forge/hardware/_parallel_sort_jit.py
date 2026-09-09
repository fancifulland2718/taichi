"""Cold, provider-specific HLSL compilation; no generic shader-search API."""

from hashlib import sha256
from pathlib import Path
import struct
import subprocess
import tempfile


def compile_shaders(compiler_path, payload, *, fused_prefix=False):
    compiler = Path(compiler_path).expanduser().resolve(strict=True)
    if not compiler.is_file():
        raise ValueError("Parallel Sort compiler_path must name the DXC executable")
    sources = Path(__file__).with_name("_parallel_sort_sources")
    source = sources / "parallel_sort.hlsl"
    facts = {
        "provider": "fidelityfx_parallel_sort",
        "upstream_commit": "0c539948c8d196ae338d91efbc8ca495f1ea0d1d",
        "source_sha256": sha256(
            source.read_bytes() + (sources / "FFX_ParallelSort.h").read_bytes()
        ).hexdigest(),
        "compiler_path": str(compiler),
        "compiler_sha256": sha256(compiler.read_bytes()).hexdigest(),
        "target": "vulkan1.1/cs_6_0",
        "tail_policy": "guarded_preloads_exact_capacity",
        "prefix_strategy": (
            "separate_histogram_fused_prefix" if fused_prefix else "reduced_scan_add"
        ),
    }
    options = {"capture_output": True, "text": True, "timeout": 120}
    version = subprocess.run([str(compiler), "--version"], check=True, **options)
    facts["compiler_version"] = version.stdout.strip()
    shaders = []
    hashes = []
    # Temporary compiler outputs never become a global cache or runtime load.
    with tempfile.TemporaryDirectory(prefix="forge-parallel-sort-") as temporary:
        entries = (
            ("Count", "Prefix", "Scatter")
            if fused_prefix
            else ("Count", "Reduce", "Scan", "ScanAdd", "Scatter")
        )
        for entry in entries:
            output = Path(temporary) / f"{entry}.spv"
            command = [
                str(compiler),
                "-spirv",
                "-fspv-target-env=vulkan1.1",
                "-fspv-entrypoint-name=main",
                "-T",
                "cs_6_0",
                "-E",
                entry,
                "-I",
                str(sources),
                str(source),
                "-Fo",
                str(output),
            ]
            if payload:
                command.append("-DkRS_ValueCopy")
            process = subprocess.run(command, **options)
            if process.returncode:
                raise RuntimeError(
                    f"Parallel Sort {entry} compilation failed: {process.stderr}"
                )
            binary = output.read_bytes()
            if len(binary) % 4 or not binary.startswith(b"\x03\x02\x23\x07"):
                raise RuntimeError(f"Parallel Sort {entry} produced invalid SPIR-V")
            shaders.append(struct.unpack(f"<{len(binary) // 4}I", binary))
            hashes.append(sha256(binary).hexdigest())
    facts["shader_sha256"] = tuple(hashes)
    return shaders, facts
