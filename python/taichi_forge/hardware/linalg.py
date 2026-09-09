"""Public optional hardware linear-algebra provider API."""

from taichi_forge.hardware._amgx import AmgxDeviceBinding, AmgxProvider, AmgxSolver
from taichi_forge.hardware._spmm_recipe import SparseSpmmRecipeProvider
from taichi_forge.hardware._matmul_recipe import MatmulRecipeProvider
from taichi_forge.hardware._sparse_solve_recipe import SparseSolveRecipeProvider
from taichi_forge.hardware._linalg import (
    CudssPlan,
    CudssRefactorSolveRecording,
    CudssSolveRecording,
    CublasGemmRecording,
    CusparseSpmmRecording,
    CusparseSpmvRecording,
    CusparseSpsmRecording,
    CusparseSpsvRecording,
    cublas_is_available,
    cusparse_is_available,
    cusparse_spmm_is_available,
    cusparse_spsm_is_available,
    cusparse_spsv_is_available,
    cudss_is_available,
    gemm_f32,
    is_available,
    spmv_f32,
    spmm_f32,
    spsm_f32,
    spsv_f32,
)

__all__ = [
    "AmgxDeviceBinding",
    "AmgxProvider",
    "AmgxSolver",
    "SparseSpmmRecipeProvider",
    "SparseSolveRecipeProvider",
    "MatmulRecipeProvider",
    "CudssPlan",
    "CudssRefactorSolveRecording",
    "CudssSolveRecording",
    "CublasGemmRecording",
    "CusparseSpmmRecording",
    "CusparseSpmvRecording",
    "CusparseSpsmRecording",
    "CusparseSpsvRecording",
    "cublas_is_available",
    "cusparse_is_available",
    "cusparse_spmm_is_available",
    "cusparse_spsm_is_available",
    "cusparse_spsv_is_available",
    "cudss_is_available",
    "gemm_f32",
    "is_available",
    "spmv_f32",
    "spmm_f32",
    "spsm_f32",
    "spsv_f32",
]
