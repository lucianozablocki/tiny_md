#!/usr/bin/env python3
import subprocess

M = 4
Ns = []
for m in range(1,M+1):
    Ns.append(4*m**3)

for N in Ns:
    print('Running: make clean')
    clean_result = subprocess.run(["make", "clean"], capture_output=True, text=True)

    if clean_result.returncode != 0:
        raise Exception("Make clean failed.")

    print(f'Running: make "CPPFLAGS="-DN={N}" ')
    make_result = subprocess.run(["make", f'CPPFLAGS="-DN={N}"'], capture_output=True, text=True)

    if make_result.returncode != 0:
        raise Exception("Make failed.")

    for _ in range(0,5):
        result = subprocess.run(["./tiny_md"], capture_output=True, text=True)
        output = result.stdout
        for line in output.split("\n"):
            if '***' in line:
                print(line)

