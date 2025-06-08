import pytest
from io import StringIO
from xdsl.context import Context
from xdsl.parser import Parser
from xdsl.utils.exceptions import ParseError

from inconspiquous.dialects import get_all_dialects
from inconspiquous.transforms.convert_qref_to_qir import (
    ConvertQrefToQir,
    QIRConversionError,
)

try:
    from inconspiquous.backend.qir import print_qir

    QIR_BACKEND_AVAILABLE = True
except ImportError:
    QIR_BACKEND_AVAILABLE = False


@pytest.fixture
def ctx():
    """Create context with all dialects registered"""
    context = Context()
    for name, dialect_factory in get_all_dialects().items():
        context.register_dialect(name, dialect_factory)
    return context


@pytest.mark.skipif(not QIR_BACKEND_AVAILABLE, reason="QIR backend not available")
def test_qir_backend_basic(ctx):
    """Test basic QIR generation"""
    program = """
    func.func @test() {
        %q = qu.alloc
        qref.gate<#gate.h> %q
        %m = qref.measure %q
        return
    }
    """

    parser = Parser(ctx, program)
    module = parser.parse_module()

    pass_instance = ConvertQrefToQir()
    pass_instance.apply(ctx, module)

    output = StringIO()
    print_qir(module, output)
    qir_output = output.getvalue()

    assert "define void @main()" in qir_output
    assert "__quantum__qis__h__body" in qir_output
    assert "__quantum__qis__mz__body" in qir_output
    assert "%Qubit" in qir_output
    assert "%Result" in qir_output


@pytest.mark.skipif(not QIR_BACKEND_AVAILABLE, reason="QIR backend not available")
def test_qir_backend_all_gates(ctx):
    """Test that different gates generate different QIR"""
    gates_to_test = ["h", "x", "y", "z", "s", "t"]

    for gate in gates_to_test:
        program = f"""
        func.func @test() {{
            %q = qu.alloc
            qref.gate<#gate.{gate}> %q
            return
        }}
        """

        parser = Parser(ctx, program)
        module = parser.parse_module()

        pass_instance = ConvertQrefToQir()
        pass_instance.apply(ctx, module)

        output = StringIO()
        print_qir(module, output)
        qir_output = output.getvalue()

        assert f"__quantum__qis__{gate}__body" in qir_output


@pytest.mark.skipif(not QIR_BACKEND_AVAILABLE, reason="QIR backend not available")
def test_qir_backend_two_qubit_gates(ctx):
    """Test two-qubit gates"""
    gates_to_test = ["cx", "cz"]
    gate_to_qir = {"cx": "cnot", "cz": "cz"}

    for gate in gates_to_test:
        program = f"""
        func.func @test() {{
            %q0 = qu.alloc
            %q1 = qu.alloc
            qref.gate<#gate.{gate}> %q0, %q1
            return
        }}
        """

        parser = Parser(ctx, program)
        module = parser.parse_module()

        pass_instance = ConvertQrefToQir()
        pass_instance.apply(ctx, module)

        output = StringIO()
        print_qir(module, output)
        qir_output = output.getvalue()

        qir_func = gate_to_qir[gate]
        assert f"__quantum__qis__{qir_func}__body" in qir_output


@pytest.mark.skipif(not QIR_BACKEND_AVAILABLE, reason="QIR backend not available")
def test_qir_backend_measurement(ctx):
    """Test measurement operations"""
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

    output = StringIO()
    print_qir(module, output)
    qir_output = output.getvalue()

    assert "__quantum__qis__mz__body" in qir_output
    assert "%Result" in qir_output


@pytest.mark.skipif(not QIR_BACKEND_AVAILABLE, reason="QIR backend not available")
def test_qir_backend_bell_state(ctx):
    """Test complete Bell state circuit"""
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

    output = StringIO()
    print_qir(module, output)
    qir_output = output.getvalue()

    assert "__quantum__qis__h__body" in qir_output
    assert "__quantum__qis__cnot__body" in qir_output
    assert "__quantum__qis__mz__body" in qir_output
    assert qir_output.count("%Result") >= 2


@pytest.mark.skipif(not QIR_BACKEND_AVAILABLE, reason="QIR backend not available")
def test_qir_backend_unsupported_gate(ctx):
    """Test handling of unsupported gates"""
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


@pytest.mark.skipif(not QIR_BACKEND_AVAILABLE, reason="QIR backend not available")
def test_qir_backend_missing_qubit(ctx):
    """Test error handling for missing qubit"""
    program = """
    func.func @test() {
        qref.gate<#gate.h> %q  # Using undefined qubit
        return
    }
    """

    parser = Parser(ctx, program)
    with pytest.raises(ParseError):
        parser.parse_module()


@pytest.mark.skipif(not QIR_BACKEND_AVAILABLE, reason="QIR backend not available")
def test_qir_backend_multiple_measurements(ctx):
    """Test multiple measurements on same qubit"""
    program = """
    func.func @test() {
        %q = qu.alloc
        %m1 = qref.measure %q
        %m2 = qref.measure %q
        return
    }
    """

    parser = Parser(ctx, program)
    module = parser.parse_module()

    pass_instance = ConvertQrefToQir()
    pass_instance.apply(ctx, module)

    output = StringIO()
    print_qir(module, output)
    qir_output = output.getvalue()

    # Accept 2 or more measurement calls (backend may emit extra for result mapping)
    assert qir_output.count("__quantum__qis__mz__body") >= 2
    assert qir_output.count("%Result") >= 2
