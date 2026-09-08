"""Compiler value proofs, not metadata-only evidence of Graph fusion."""

import json

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core
from taichi_forge.graph._graph import gen_cpp_kernel
from tests import test_utils


@ti.kernel
def _pointwise(
    left: ti.types.ndarray(),
    right: ti.types.ndarray(),
    output: ti.types.ndarray(),
    factor: ti.u32,
):
    for i in output:
        bits = ti.cast(left[i], ti.u32) * factor + ti.u32(0xF0000001)
        bits = (bits ^ ti.cast(right[i], ti.u32)) - ti.cast(i, ti.u32)
        output[i] = (bits << 3) | (bits >> 29)


def _query(kernel, arguments):
    return core._graph_pointwise_value_program(gen_cpp_kernel(kernel, arguments))


def _array(name, dtype=ti.i32):
    return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=1)


def _evaluate_value(program, arguments, index):
    # Independent bit-vector evaluator: every supported operation produces
    # exactly 32 bits, with explicit signed arithmetic-right-shift semantics.
    values = []
    mask = 0xFFFFFFFF
    for number, node in enumerate(program["nodes"]):
        assert node["dtype"] in ("i32", "u32")
        assert all(0 <= operand < number for operand in node["operands"])
        operands = [values[operand] for operand in node["operands"]]
        kind, operation = node["kind"], node["operation"]
        if kind == "constant":
            value = node["constant_bits"]
        elif kind == "index":
            value = index
        elif kind == "array_load":
            value = int(arguments[node["argument"]][index])
        elif kind == "scalar_argument":
            value = int(arguments[node["argument"]])
        elif kind == "unary":
            value = {
                "cast_value": lambda: operands[0],
                "cast_bits": lambda: operands[0],
                "neg": lambda: -operands[0],
                "bit_not": lambda: ~operands[0],
            }[operation]()
        else:
            a, b = operands
            signed_a = a if a < 0x80000000 else a - 0x100000000
            arithmetic_a = (
                signed_a
                if program["nodes"][node["operands"][0]]["dtype"] == "i32"
                else a
            )
            value = {
                "add": lambda: a + b,
                "sub": lambda: a - b,
                "mul": lambda: a * b,
                "bit_and": lambda: a & b,
                "bit_or": lambda: a | b,
                "bit_xor": lambda: a ^ b,
                "bit_shl": lambda: a << b,
                "bit_shr": lambda: a >> b,
                "bit_sar": lambda: arithmetic_a >> b,
            }[operation]()
        values.append(value & mask)
    return values[program["result"]]


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("dtype", (ti.i32, ti.u32))
def test_pointwise_value_dag_matches_integer_bits_without_mutating_kernel(dtype):
    arguments = [_array(name, dtype) for name in ("left", "right", "output")]
    arguments.append(ti.graph.Arg(ti.graph.ArgKind.SCALAR, "factor", ti.u32))
    raw = np.asarray((0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF), np.uint32)
    left = raw if dtype == ti.u32 else raw.view(np.int32)
    right = left[::-1].copy()
    arrays = [ti.ndarray(dtype, shape=len(raw)) for _ in range(3)]
    arrays[0].from_numpy(left)
    arrays[1].from_numpy(right)
    factor = 0x60000001
    _pointwise(*arrays, factor)
    original = arrays[2].to_numpy().view(np.uint32)
    arrays[2].fill(42)
    program = _query(_pointwise, arguments)
    assert program["available"], program
    assert program["output_argument"] == 2
    assert program["metadata"]["iteration_domain"]["kind"] == "external_tensor"
    assert program == json.loads(json.dumps(_query(_pointwise, arguments)))
    assert 0 < len(program["nodes"]) <= 64
    np.testing.assert_array_equal(arrays[2].to_numpy(), 42)
    expected = []
    for i in range(len(raw)):
        bits = ((int(raw[i]) * factor + 0xF0000001) & 0xFFFFFFFF) ^ int(raw[-1 - i])
        bits = (bits - i) & 0xFFFFFFFF
        expected.append(((bits << 3) | (bits >> 29)) & 0xFFFFFFFF)
    np.testing.assert_array_equal(original, expected)
    np.testing.assert_array_equal(
        [
            _evaluate_value(program, [left, right, None, factor], i)
            for i in range(len(raw))
        ],
        expected,
    )
    _pointwise(*arrays, factor)
    np.testing.assert_array_equal(arrays[2].to_numpy().view(np.uint32), expected)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_pointwise_value_domain_and_signed_shift_remain_explicit():
    @ti.kernel
    def bounded(
        source: ti.types.ndarray(dtype=ti.i32),
        output: ti.types.ndarray(dtype=ti.i32),
        count: ti.i32,
    ):
        for i in range(count):
            output[i] = (~source[i]) >> 3

    arguments = [
        _array("source"),
        _array("output"),
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "count", ti.i32),
    ]
    program = _query(bounded, arguments)
    assert program["available"], program
    domain = program["metadata"]["iteration_domain"]
    assert domain["kind"] == "scalar_argument" and domain["arg_id"] == [2]
    raw = np.asarray((0, -1, 0x7FFFFFFF, -0x80000000), np.int32)
    source, output = (ti.ndarray(ti.i32, shape=4) for _ in range(2))
    source.from_numpy(raw)
    output.fill(123)
    bounded(source, output, 3)
    expected = np.asarray([(~int(value)) >> 3 for value in raw[:3]], np.int32)
    np.testing.assert_array_equal(output.to_numpy(), np.append(expected, 123))
    np.testing.assert_array_equal(
        [_evaluate_value(program, [raw, None, 3], i) for i in range(3)],
        expected.view(np.uint32),
    )
    # The proof records the domain; it does not claim coverage of output[3].


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_pointwise_value_rejects_nonrepresentable_semantics_without_partial_dag():
    @ti.kernel
    def divided(a: ti.types.ndarray(), b: ti.types.ndarray(), divisor: ti.i32):
        for i in b:
            b[i] = a[i] // divisor

    @ti.kernel
    def shifted(a: ti.types.ndarray(), b: ti.types.ndarray(), count: ti.i32):
        for i in b:
            b[i] = a[i] << count

    @ti.kernel
    def gathered(a: ti.types.ndarray(), b: ti.types.ndarray()):
        for i in b:
            b[i] = a[a[i]]

    @ti.kernel
    def conditional(a: ti.types.ndarray(), b: ti.types.ndarray()):
        for i in b:
            if a[i] > 0:
                b[i] = a[i]

    @ti.kernel
    def stores(a: ti.types.ndarray(), b: ti.types.ndarray()):
        for i in b:
            b[i] = a[i] * 2
            a[i] = a[i] + 1

    @ti.kernel
    def randomized(a: ti.types.ndarray(), b: ti.types.ndarray()):
        for i in b:
            b[i] = a[i] + ti.random(ti.i32)

    base = [_array("a"), _array("b")]
    scalar = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "scalar", ti.i32)
    for kernel, arguments in (
        (divided, [*base, scalar]),
        (shifted, [*base, scalar]),
        (gathered, base),
        (conditional, base),
        (stores, base),
        (randomized, base),
        (stores, [_array("a", ti.f32), _array("b", ti.f32)]),
    ):
        program = _query(kernel, arguments)
        assert not program["available"], kernel.__name__
        assert program["blocker"] and not program["nodes"]
        assert program["result"] == program["output_argument"] == -1
