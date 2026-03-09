import pytest
from xdsl.dialects.builtin import i1
from xdsl.ir import TypeAttribute, VerifyException
from xdsl.irdl import (
    AnyAttr,
    AnyInt,
    ConstraintContext,
    EqAttrConstraint,
    EqIntConstraint,
    RangeOf,
    SingleOf,
    irdl_attr_definition,
)

from inconspiquous.dialects.instrument import (
    ConstantInstrumentOp,
    InstrumentAttr,
    InstrumentConstraint,
    InstrumentType,
)


@irdl_attr_definition
class DummyInstrument(InstrumentAttr):
    name = "test.dummy"

    @property
    def num_qubits(self) -> int:
        return 1

    @property
    def classical_results(self) -> tuple[TypeAttribute, ...]:
        return ()


@irdl_attr_definition
class WithResultsInstrument(InstrumentAttr):
    name = "test.results"

    @property
    def num_qubits(self) -> int:
        return 2

    @property
    def classical_results(self) -> tuple[TypeAttribute, ...]:
        return (i1, i1)


def test_constant_instrument_init():
    dummy_op = ConstantInstrumentOp(DummyInstrument())

    dummy_op.verify()
    assert dummy_op.result_types[0] == InstrumentType(1)

    result_op = ConstantInstrumentOp(WithResultsInstrument())

    result_op.verify()
    assert result_op.result_types[0] == InstrumentType(2, (i1, i1))


def test_instrument_constraint():

    constraint_dummy = InstrumentConstraint(
        EqIntConstraint(1), RangeOf(AnyAttr()).of_length(EqIntConstraint(0))
    )

    constraint_dummy.verify(DummyInstrument(), ConstraintContext())

    constraint_qubits = InstrumentConstraint(EqIntConstraint(3), RangeOf(AnyAttr()))

    with pytest.raises(VerifyException, match="Invalid value 1, expected 3"):
        constraint_qubits.verify(DummyInstrument(), ConstraintContext())

    constraint_result = InstrumentConstraint(AnyInt(), SingleOf(EqAttrConstraint(i1)))

    with pytest.raises(VerifyException, match="Expected a single attribute, got 0"):
        constraint_result.verify(DummyInstrument(), ConstraintContext())

    constraint_with_result = InstrumentConstraint(
        EqIntConstraint(2), RangeOf(EqAttrConstraint(i1)).of_length(EqIntConstraint(2))
    )

    constraint_with_result.verify(WithResultsInstrument(), ConstraintContext())

    with pytest.raises(VerifyException, match="Invalid value 2, expected 3"):
        constraint_qubits.verify(WithResultsInstrument(), ConstraintContext())

    constraint_result = InstrumentConstraint(AnyInt(), SingleOf(EqAttrConstraint(i1)))

    with pytest.raises(VerifyException, match="Expected a single attribute, got 2"):
        constraint_result.verify(WithResultsInstrument(), ConstraintContext())
