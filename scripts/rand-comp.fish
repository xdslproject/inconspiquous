set iterations 1 2 4 8 16 32 64 100 200 300 400 500 600 700 800 900 1000
set this_folder (dirname (status -f))

for x in $iterations
  echo -n $x
  echo -n ,
  set t (uv run $this_folder/rand-comp-naive.py $x | string collect)
  echo -n $t
  echo -n ,
  echo Naive

  echo -n $x
  echo -n ,
  set t (uv run $this_folder/rand-comp-dg.py $x | string collect)
  echo -n $t
  echo -n ,
  echo Dynamic gates
end
