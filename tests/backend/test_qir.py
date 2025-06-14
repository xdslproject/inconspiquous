import pytest

from xdsl.dialects.builtin import ModuleOp

from inconspiquous.backend.qir import pyqir_available, print_qir
from inconspiquous.dialects import qir

# Skip tests if PyQIR is not available
_qir_backend_available = pyqir_available


@pytest.fixture
def ctx():
    """Create a context for testing"""
    from xdsl.ir import Context

    return Context()


def test_qir_backend_basic(ctx):
    """Test basic QIR generation"""
    if not _qir_backend_available:
        pytest.skip("PyQIR not available")

    # Create a simple module with one qubit and one measurement
    module = ModuleOp.from_region_or_ops([])
    func = qir.FuncOp.from_region_or_ops(
        "main",
        [],
        [qir.QubitAllocateOp(), qir.MeasureOp()],
    )
    module.body.blocks[0].add_op(func)

    # Convert to QIR
    qir_module = print_qir(module)
    assert qir_module is not None
    assert "main" in qir_module


def test_qir_backend_all_gates(ctx):
    """Test all QIR gates"""
    if not _qir_backend_available:
        pytest.skip("PyQIR not available")

    # Create a module with all gates
    module = ModuleOp.from_region_or_ops([])
    func = qir.FuncOp.from_region_or_ops(
        "main",
        [],
        [
            qir.QubitAllocateOp(),  # q0
            qir.QubitAllocateOp(),  # q1
            qir.HGateOp(),  # H q0
            qir.XGateOp(),  # X q0
            qir.YGateOp(),  # Y q0
            qir.ZGateOp(),  # Z q0
            qir.SGateOp(),  # S q0
            qir.TGateOp(),  # T q0
            qir.CXGateOp(),  # CX q0, q1
            qir.CZGateOp(),  # CZ q0, q1
            qir.MeasureOp(),  # Measure q0
        ],
    )
    module.body.blocks[0].add_op(func)

    # Convert to QIR
    qir_module = print_qir(module)
    assert qir_module is not None
    assert "main" in qir_module
    assert "h" in qir_module
    assert "x" in qir_module
    assert "y" in qir_module
    assert "z" in qir_module
    assert "s" in qir_module
    assert "t" in qir_module
    assert "cx" in qir_module
    assert "cz" in qir_module
    assert "mz" in qir_module
