#!/bin/bash

test_path=test

echo "== COMPILE"
execfile=solution

# compilation stage
make clean  > /dev/null
make > /dev/null

if [[ $? -eq 0 && `find "$execfile"` ]]; then
    # if compilation OK and exec file exist, run test
    echo == EXECUTE TESTS
    echo ==

    # execution time
    START=$(date +%s.%N)
    
    for test_file in `ls $test_path/*.quest`; do
        test_name=$(basename $test_file .quest)
        solve_file=$test_path/$test_name.solv
        answer_file=$test_path/$test_name.answ

        echo == RUNNING "< $test_name >"

        "./$execfile" < $test_file > $solve_file

        if ! diff -w -u $answer_file  $solve_file >> /dev/null; then
            echo == TEST $test_index FAIL
            diff --color -u $answer_file  $solve_file
        else
            echo == TEST $test_index OK
        fi
    done
    END=$(date +%s.%N)
    DIFF=$(echo "$END - $START" | bc)
    echo == EXEC TIME $DIFF c
else
    echo == FAIL BUILD
fi
echo ==
