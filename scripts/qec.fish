set passes convert-to-xzs xzs-select xz-commute canonicalize cse
set pass_names convert_to_xzs xzs_select xz_commute canonicalize

set convert_to_xzs
set xzs_select
set xz_commute
set canonicalize

set files (dirname (status -f))/qec/prog-*.mlir

for file in $files
    echo Testing $file >&2
    for pass in $pass_names
        set -a $pass 100
    end
    for i in (seq 5)
    	echo iteration $i/5 >&2
    	set output (uv run quopt $file --time-passes -p convert-qref-to-qssa,qec-inline,convert-to-xzs,xzs-select,xz-commute,canonicalize | grep 'Pass' | string collect)
	for j in (seq 4)
	    set $pass_names[$j][-1] (math min $$pass_names[$j][-1] , (echo $output | grep "$passes[$j]" | grep -o '[[:digit:]][[:digit:]]*\.[[:digit:]][[:digit:]]*'))
	end
    end
end

echo Collecting results >&2
echo "Cycles,Runtime (s),Pass"
for j in (seq 4)
    for i in (seq (count $files))
    	echo -n (math 100 x $i)
    	echo -n ,
    	echo -n $$pass_names[$j][$i]
	echo -n ,
	echo $passes[$j]
    end
end
