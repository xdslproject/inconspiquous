import unittest
from inconspiquous.dialects.qref import GateOp, MeasureOp as QrefMeasureOp
from inconspiquous.dialects.qir import HOp, MeasureOp as QirMeasureOp
from inconspiquous.transforms.convert_qref_to_qir import convert_qref_to_qir

class DummyQrefModule:
    def __init__(self, ops):
        self.ops = ops

class TestConvertQrefToQir(unittest.TestCase):
    def test_simple_gate_and_measure(self):
        qref_module = DummyQrefModule([
            GateOp(None),  # GateAttr is None for stub
            QrefMeasureOp(),
        ])
        qir_module = convert_qref_to_qir(qref_module)
        self.assertEqual(len(qir_module.ops), 2)
        self.assertIsInstance(qir_module.ops[0], HOp)
        self.assertIsInstance(qir_module.ops[1], QirMeasureOp)

if __name__ == "__main__":
    unittest.main()
