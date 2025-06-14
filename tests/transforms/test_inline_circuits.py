"""
Test for the inline circuits transformation.
"""

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block, Region

from inconspiquous.dialects.qssa import CircuitOp, ReturnOp, GateOp, DynGateOp
from inconspiquous.dialects.qu import BitType, AllocOp
from inconspiquous.dialects.gate import XGate, YGate, ZGate
from inconspiquous.transforms.inline_circuits import InlineCircuitsPass


def test_basic_circuit_creation():
    """Test basic import and pass registration."""

    # Just test that the pass can be instantiated and the dialects work
    pass_instance = InlineCircuitsPass()
    assert pass_instance.name == "inline-circuits"

    # Test that we can create circuit operations
    circuit_block = Block(arg_types=[BitType(), BitType()])

    # Add return operation
    return_op = ReturnOp(circuit_block.args[0], circuit_block.args[1])
    circuit_block.add_op(return_op)

    # Create circuit operation
    circuit_region = Region([circuit_block])
    circuit = CircuitOp(2, circuit_region)

    # Verify circuit has the right properties
    assert len(circuit.body.blocks) == 1
    assert len(circuit.body.blocks[0].args) == 2


def test_circuit_inlining():
    """Test actual circuit inlining functionality."""

    # Create a circuit that applies X gate to its input
    circuit_block = Block(arg_types=[BitType()])

    # Add X gate operation inside the circuit
    x_gate = GateOp(XGate(), circuit_block.args[0])
    circuit_block.add_op(x_gate)

    # Return the result of the X gate
    return_op = ReturnOp(x_gate.results[0])
    circuit_block.add_op(return_op)

    # Create circuit operation
    circuit_region = Region([circuit_block])
    circuit = CircuitOp(1, circuit_region)

    # Create an input qubit using allocation
    input_alloc = AllocOp()  # This creates a qubit

    # Create dyn_gate operation using the circuit
    dyn_gate = DynGateOp(circuit.result, input_alloc.results[0])

    # Create module with all operations
    module = ModuleOp([input_alloc, circuit, dyn_gate])

    # Apply the inline circuits pass
    pass_instance = InlineCircuitsPass()
    pass_instance.apply(Context(), module)

    # Count operations after inlining
    ops_after = list(module.body.ops)

    # Verify that inlining happened correctly
    gate_ops = [op for op in ops_after if isinstance(op, GateOp)]
    dyn_gate_ops = [op for op in ops_after if isinstance(op, DynGateOp)]
    alloc_ops = [op for op in ops_after if isinstance(op, AllocOp)]

    # After inlining, we should have:
    # - Original allocation operation (1 alloc)
    # - Inlined circuit operation (1 gate: X)
    # - No dyn_gate operations (should be replaced)
    # - Circuit operation should still exist

    assert len(alloc_ops) == 1, f"Expected 1 alloc operation, got {len(alloc_ops)}"
    assert len(gate_ops) >= 1, (
        f"Expected at least 1 gate operation after inlining, got {len(gate_ops)}"
    )
    assert len(dyn_gate_ops) == 0, (
        f"Expected no dyn_gate operations after inlining, got {len(dyn_gate_ops)}"
    )


def test_empty_circuit():
    """Test circuit with no operations (just return)."""

    # Create a circuit that just returns its inputs unchanged
    circuit_block = Block(arg_types=[BitType()])
    return_op = ReturnOp(circuit_block.args[0])
    circuit_block.add_op(return_op)

    circuit_region = Region([circuit_block])
    circuit = CircuitOp(1, circuit_region)

    # Verify the circuit is valid
    assert len(circuit.body.blocks) == 1
    assert len(circuit.body.blocks[0].args) == 1
    assert len(circuit.body.blocks[0].ops) == 1  # Just the return


def test_complex_circuit_inlining():
    """Test inlining of a circuit with multiple operations."""

    # Create a circuit that applies X then Z gates to its input
    circuit_block = Block(arg_types=[BitType()])

    # Add X gate operation inside the circuit
    x_gate = GateOp(XGate(), circuit_block.args[0])
    circuit_block.add_op(x_gate)

    # Add Z gate operation on the result of X gate
    z_gate = GateOp(ZGate(), x_gate.results[0])
    circuit_block.add_op(z_gate)

    # Return the result of the Z gate
    return_op = ReturnOp(z_gate.results[0])
    circuit_block.add_op(return_op)

    # Create circuit operation
    circuit_region = Region([circuit_block])
    circuit = CircuitOp(1, circuit_region)

    # Create input qubits
    input_alloc = AllocOp()

    # Create dyn_gate operation using the circuit
    dyn_gate = DynGateOp(circuit.result, input_alloc.results[0])

    # Create module with all operations
    module = ModuleOp([input_alloc, circuit, dyn_gate])

    # Apply the inline circuits pass
    pass_instance = InlineCircuitsPass()
    pass_instance.apply(Context(), module)

    # Count operations after inlining
    ops_after = list(module.body.ops)

    # Verify that inlining happened correctly
    gate_ops = [op for op in ops_after if isinstance(op, GateOp)]
    dyn_gate_ops = [op for op in ops_after if isinstance(op, DynGateOp)]

    # After inlining, we should have:
    # - 2 gate operations (X and Z from the circuit)
    # - 0 dyn_gate operations

    assert len(gate_ops) == 2, (
        f"Expected exactly 2 gate operations after inlining, got {len(gate_ops)}"
    )
    assert len(dyn_gate_ops) == 0, (
        f"Expected no dyn_gate operations after inlining, got {len(dyn_gate_ops)}"
    )

    # Verify the gate types are correct
    gate_types = [type(op.gate) for op in gate_ops]
    assert XGate in gate_types, "Expected X gate to be inlined"
    assert ZGate in gate_types, "Expected Z gate to be inlined"


def test_two_qubit_circuit_inlining():
    """Test inlining of a two-qubit circuit."""

    # Create a circuit that applies X to first qubit and Y to second qubit
    circuit_block = Block(arg_types=[BitType(), BitType()])

    # Add X gate to first qubit
    x_gate = GateOp(XGate(), circuit_block.args[0])
    circuit_block.add_op(x_gate)

    # Add Y gate to second qubit
    y_gate = GateOp(YGate(), circuit_block.args[1])
    circuit_block.add_op(y_gate)

    # Return both results
    return_op = ReturnOp(x_gate.results[0], y_gate.results[0])
    circuit_block.add_op(return_op)

    # Create circuit operation
    circuit_region = Region([circuit_block])
    circuit = CircuitOp(2, circuit_region)

    # Create input qubits
    input_alloc1 = AllocOp()
    input_alloc2 = AllocOp()

    # Create dyn_gate operation using the circuit
    dyn_gate = DynGateOp(
        circuit.results[0], input_alloc1.results[0], input_alloc2.results[0]
    )

    # Create module with all operations
    module = ModuleOp([input_alloc1, input_alloc2, circuit, dyn_gate])

    # Apply the inline circuits pass
    pass_instance = InlineCircuitsPass()
    pass_instance.apply(Context(), module)

    # Verify that inlining happened correctly
    ops_after = list(module.body.ops)
    gate_ops = [op for op in ops_after if isinstance(op, GateOp)]
    dyn_gate_ops = [op for op in ops_after if isinstance(op, DynGateOp)]

    assert len(gate_ops) == 2, (
        f"Expected exactly 2 gate operations after inlining, got {len(gate_ops)}"
    )
    assert len(dyn_gate_ops) == 0, (
        f"Expected no dyn_gate operations after inlining, got {len(dyn_gate_ops)}"
    )
