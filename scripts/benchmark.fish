set benches \
  Teleport,teleport \
  Prep,logical_plus \
  RUS,repeat_until_success \
  QML,qml \
  IPE,ipe \
  RWPE,rwpe \
  MBQC-CX,mbqc_cx \
  MBQC-ROT,mbqc_rotation \
  QEC-Adap,adaptive_qec \
  QAOA,qaoa

echo Benchmark,Line count,Word count,Basic blocks,Cyclomatic complexity,Quantum gates,Halstead difficulty,IR
for x in $benches
    echo $x | read -d , name file
    echo -n $name
    echo -n ,
    set file (dirname (status -f))/../tests/filecheck/bench/$file.mlir
    set dg (uv run quopt $file | string collect)
    # Line count
    echo $dg | grep -v '^\($\|!\|;\)' | wc -l | string collect

    echo -n ,

    # Word count
    echo $dg | grep -v '^\($\|!\|;\)' | wc -w | string collect

    echo -n ,

    # Basic blocks
    echo $dg | grep -v '^\(builtin.module\|//\)' | grep '{' | wc -l | string collect

    echo -n ,

    echo $dg | grep 'scf.if\|scf.while\|func.func\|scf.for' | wc -l | string collect

    echo -n ,

    echo $dg | grep 'qssa\.' | wc -l | string collect

    echo -n ,

    set operands (echo $dg | grep -o '%[a-z0-9][a-z0-9_]*' | wc -l)
    set unique_operands (echo $dg | grep -o '%[a-z0-9][a-z0-9_]*' | sort | uniq | wc -l)
    set unique_ops (echo $dg | grep '[a-z]*\.[a-z]*' | sed 's/^\s*//g' | sed 's/%[a-z0-9][a-z0-9_]*//g' | sort | uniq | wc -l)
    math $unique_ops / 2 + \($operands - $unique_operands\) / $unique_operands | string collect

    echo -n ,

    echo "Dynamic gate"

    set qir (uv run quopt $file -p convert-qssa-to-qref,lower-xzs-to-select,cse,canonicalize,lower-dyn-gate-to-scf,canonicalize,convert-qref-to-qir,convert-qir-to-llvm | mlir-opt -p 'builtin.module(convert-scf-to-cf,canonicalize,convert-math-to-llvm,convert-arith-to-llvm,convert-cf-to-llvm,convert-func-to-llvm)' | mlir-translate --mlir-to-llvmir | string collect)

    echo -n $name

    echo -n ,

    echo $qir | grep -v '^\($\|!\|;\)' | wc -l | string collect

    echo -n ,

    echo $qir | grep -v '^\($\|!\|;\)' | wc -w | string collect

    echo -n ,

    echo $qir | grep '[[:digit:]][[:digit:]]*:\|define' | wc -l | string collect

    echo -n ,

    math (echo $qir | grep -o 'define\|label' | wc -l) - (echo $qir | grep '[[:digit:]][[:digit:]]*:' | wc -l) | string collect

    echo -n ,

    echo $qir | grep 'call.*@__quantum__qis' | wc -l | string collect

    echo -n ,

    set qir_operands (echo $qir | grep -o '%[a-z0-9][a-z0-9_]*' | wc -l)
    set qir_unique_operands (echo $qir | grep -o '%[a-z0-9][a-z0-9_]*' | sort | uniq | wc -l)
    set qir_unique_ops (echo $qir | grep -v '^declare\|^source\|^;\|^!\|^[0-9]\|^\}' | sed 's/^\s*//g' | sed 's/%[a-z0-9][a-z0-9_]*//g' | sort | uniq | wc -l)
    math $qir_unique_ops / 2 + \($qir_operands - $qir_unique_operands\) / $qir_unique_operands | string collect

    echo -n ,

    echo "QIR"
end
