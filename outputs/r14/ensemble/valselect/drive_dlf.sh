#!/bin/zsh
cd <artifact-root>
export PYTHONPATH=src
D=outputs/r14/ensemble/valselect
run_one() {
  set -- $1
  if [ -f "$D/preds/$1/$2__seed$3.json" ]; then echo "SKIP $1 $2 $3"; return; fi
  nice -n 15 python3 $D/run_dlf.py $1 $2 $3 >> $D/logs/dlf_$1_$2_$3.log 2>&1 \
    && echo "OK $1 $2 $3" || echo "FAIL $1 $2 $3"
}
export -f run_one 2>/dev/null
cat $D/dlf_jobs.txt | xargs -P 6 -I{} zsh -c 'cd <artifact-root>; export PYTHONPATH=src; D=outputs/r14/ensemble/valselect; set -- {}; if [ -f "$D/preds/$1/$2__seed$3.json" ]; then echo "SKIP $1 $2 $3"; else nice -n 15 python3 $D/run_dlf.py $1 $2 $3 >> $D/logs/dlf_$1_$2_$3.log 2>&1 && echo "OK $1 $2 $3" || echo "FAIL $1 $2 $3"; fi'
echo "DLF QUEUE DONE"
