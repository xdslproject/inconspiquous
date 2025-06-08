import pytest
from xdsl.context import Context
from xdsl.parser import Parser

from inconspiquous.dialects import get_all_dialects
from inconspiquous.transforms.convert_qref_to_qir import (
    ConvertQrefToQir,
    QIRConversionError,
)


@pytest.fixture
def ctx():
    """Create context with all dialects registered"""
    context = Context()
    for name, dialect_factory in get_all_dialects().items():
        context.register_dialect(name, dialect_factory)
    return context


def test_simple_gate_conversion(ctx):
    """Test converting a simple gate"""
    program = """
    func.func @test() {
        %q = qu.alloc
        qref.gate<#gate.h> %q
        return
    }
    """

    parser = Parser(ctx, program)
    module = parser.parse_module()

    pass_instance = ConvertQrefToQir()
    pass_instance.apply(ctx, module)

    ops = list(module.walk())
    qir_ops = [op for op in ops if op.name.startswith("qir.")]

    assert len(qir_ops) >= 2  # Should have at least alloc + gate

    op_names = [op.name for op in qir_ops]
    assert "qir.qubit_allocate" in op_names
    assert "qir.h" in op_names


def test_all_single_qubit_gates(ctx):
    """Test converting all single qubit gates"""
    gates = ["h", "x", "y", "z", "s", "t"]

    for gate_name in gates:
        program = f"""
        func.func @test() {{
            %q = qu.alloc
            qref.gate<#gate.{gate_name}> %q
            return
        }}
        """

        parser = Parser(ctx, program)
        module = parser.parse_module()

        pass_instance = ConvertQrefToQir()
        pass_instance.apply(ctx, module)

        ops = list(module.walk())
        op_names = [op.name for op in ops]

        assert "qir.qubit_allocate" in op_names
        assert f"qir.{gate_name}" in op_names


def test_two_qubit_gates(ctx):
    """Test converting two-qubit gates"""
    gates = ["cx", "cz"]

    for gate_name in gates:
        program = f"""
        func.func @test() {{
            %q0 = qu.alloc
            %q1 = qu.alloc
            qref.gate<#gate.{gate_name}> %q0, %q1
            return
        }}
        """

        parser = Parser(ctx, program)
        module = parser.parse_module()

        pass_instance = ConvertQrefToQir()
        pass_instance.apply(ctx, module)

        ops = list(module.walk())
        op_names = [op.name for op in ops]

        assert f"qir.{gate_name}" in op_names


def test_measurement_conversion(ctx):
    """Test converting measurements"""
    program = """
    func.func @test() {
        %q = qu.alloc
        %m = qref.measure %q
        return
    }
    """

    parser = Parser(ctx, program)
    module = parser.parse_module()

    pass_instance = ConvertQrefToQir()
    pass_instance.apply(ctx, module)

    ops = list(module.walk())
    op_names = [op.name for op in ops]

    assert "qir.qubit_allocate" in op_names
    assert "qir.measure" in op_names
    assert "qir.read_result" in op_names


def test_bell_state_circuit(ctx):
    """Test converting a complete Bell state circuit"""
    program = """
    func.func @bell_state() {
        %q0 = qu.alloc
        %q1 = qu.alloc
        
        qref.gate<#gate.h> %q0
        qref.gate<#gate.cx> %q0, %q1
        
        %m0 = qref.measure %q0
        %m1 = qref.measure %q1
        
        return
    }
    """

    parser = Parser(ctx, program)
    module = parser.parse_module()

    pass_instance = ConvertQrefToQir()
    pass_instance.apply(ctx, module)

    ops = list(module.walk())
    op_names = [op.name for op in ops]

    # Should have 2 allocations
    assert op_names.count("qir.qubit_allocate") == 2

    # Should have H and CX gates
    assert "qir.h" in op_names
    assert "qir.cx" in op_names

    # Should have 2 measurements
    assert op_names.count("qir.measure") == 2
    assert op_names.count("qir.read_result") == 2


def test_unsupported_gates_skipped(ctx):
    """Test that unsupported gates raise QIRConversionError"""
    program = """
    func.func @test() {
        %q0 = qu.alloc
        %q1 = qu.alloc
        %q2 = qu.alloc
        qref.gate<#gate.toffoli> %q0, %q1, %q2
        return
    }
    """

    parser = Parser(ctx, program)
    module = parser.parse_module()

    pass_instance = ConvertQrefToQir()
    with pytest.raises(QIRConversionError):
        pass_instance.apply(ctx, module)
