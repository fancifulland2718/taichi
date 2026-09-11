"""Runtime-bound and sparse linear algebra support."""

from taichi_forge.linalg import experimental
from taichi_forge.linalg._fft import record_fft
from taichi_forge.linalg._matmul import record_matmul
from taichi_forge.linalg._contraction import record_contraction
from taichi_forge.linalg._sparse_matmul import record_sparse_matmul
from taichi_forge.linalg._sparse_solve import record_sparse_solve
from taichi_forge.linalg._prepared_operator import PreparedOperatorPlan
from taichi_forge.linalg._runtime import (
    LinearOperator,
    OperatorCapabilities,
    OperatorQualificationReport,
    OperatorTraits,
    SmallBlockInverseBuilder,
    SmallBlockInverseResult,
    VectorView,
    aslinearoperator,
    block_diagonal,
    identity,
    inverse_block_diagonal,
    qualify_operator,
    summarize_operator_qualifications,
    vector_io_capabilities,
    vector_view,
)
from taichi_forge.linalg.matrixfree_cg import (
    FieldLinearOperator,
    MatrixFreeBICGSTAB,
    MatrixFreeCG,
)
from taichi_forge.linalg.sparse_bicgstab import SparseBiCGSTAB
from taichi_forge.linalg.sparse_cg import SparseCG
from taichi_forge.linalg.sparse_minres import SparseMINRES
from taichi_forge.linalg.sparse_matrix import *
from taichi_forge.linalg.sparse_solver import SparseSolver
