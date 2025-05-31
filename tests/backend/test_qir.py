import pytest
from io import StringIO
from xdsl.context import Context
from xdsl.parser import Parser

from inconspiquous.dialects import get_all_dialects
from inconspiquous.transforms.convert_qref_to_qir import ConvertQrefToQir

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
