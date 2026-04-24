"""P3.d correctness: the get_pos_info fast-path must still produce intelligible
error messages that include the offending source line and a caret underline
pointing at the node.

We trigger three representative compile errors and verify the rendered message
has the key properties:
  1. contains the expected source fragment, and
  2. contains at least one '^' caret, and
  3. the caret line is correctly aligned (caret length > 0, caret line follows
     the source line in output order).
"""
from __future__ import annotations
import taichi as ti


def _assert_error_format(src_fragment: str, trigger):
    ti.init(arch=ti.cpu)
    try:
        trigger()
    except Exception as e:  # noqa: BLE001
        msg = str(e)
    else:
        raise AssertionError(f"expected compile error for fragment {src_fragment!r}")
    assert src_fragment in msg, f"missing source fragment in error.\nmsg=\n{msg}"
    assert "^" in msg, f"missing caret in error.\nmsg=\n{msg}"
    # Check caret follows source: find the source line, then look for ^ afterwards
    i = msg.find(src_fragment)
    tail = msg[i + len(src_fragment):]
    assert "^" in tail, f"caret should appear after source fragment.\nmsg=\n{msg}"
    print(f"  OK fragment={src_fragment!r}")
    ti.reset()


def test_name_error():
    def go():
        @ti.kernel
        def k():
            x = undeclared_symbol  # noqa: F821
        k()
    _assert_error_format("undeclared_symbol", go)


def test_bad_subscript():
    def go():
        f = ti.field(ti.f32, shape=4)
        @ti.kernel
        def k():
            a = f[1, 2, 3]  # field is 1D, subscript with 3 indices
        k()
    _assert_error_format("f[1, 2, 3]", go)


def test_type_error_in_op():
    def go():
        @ti.kernel
        def k():
            # Use an undefined name inside a binop so AST walks in
            x = 1 + totally_undefined_name  # noqa: F821
        k()
    _assert_error_format("totally_undefined_name", go)


def main():
    test_name_error()
    test_bad_subscript()
    test_type_error_in_op()
    print("OK")


if __name__ == "__main__":
    main()
