#!/usr/bin/env python3
import subprocess
import csv
import datetime
import os

os.makedirs('results', exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

M = 4
Ns = []
for m in range(1,M+1):
    Ns.append(4*m**3)

opt_flags = ['-O0', '-O1', '-O2', '-O3'] # Podriamos agregar fast math
compilers = ['gcc', 'clang']
results = []

for compiler in compilers:
    for opt in opt_flags:
        for N in Ns:
            print('Running: make clean')
            clean_result = subprocess.run(["make", "clean"], capture_output=True, text=True)

            if clean_result.returncode != 0:
                raise Exception("Make clean failed.")

            print(f'Running: make CC="{compiler}" CPPFLAGS="-DN={N}" CFLAGS="{opt}"')
            make_result = subprocess.run(["make", f'CC="{compiler}"', f'CPPFLAGS="-DN={N}"', f'CFLAGS="{opt}"'], capture_output=True, text=True)

            if make_result.returncode != 0:
                raise Exception("Make failed.")

            for _ in range(0,5):
                result = subprocess.run(["./tiny_md"], capture_output=True, text=True)
                output = result.stdout
                for line in output.split("\n"):
                    if '***' in line:
                        particulas_s = f'{
                            float(
                                line.split('***')[1] # sacar asteriscos
                                .replace('particulas/s:', "") # sacar el string "particulas/s:"
                                .strip() # remover cualquier leading/trailing whitespace
                                ):.3f
                            }'
                        results.append((particulas_s, N, opt, compiler))
                        print(line)

with open(f'results/{timestamp}.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['particulas/s', 'N', 'opt', 'compiler'])
    writer.writerows(results)
