#!/usr/bin/env python3
import subprocess
import csv
import datetime
import os
import logging

os.makedirs('results', exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s.%(lineno)d - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler(os.path.join('results', f'{timestamp}log.txt'), mode='w'),
    ]
)
logger = logging.getLogger(__name__)

M = 6
Ns = []
for m in range(1,M+1):
    Ns.append(4*m**3)

runs = [30,25,20,15,10,5]
opt_flags = ['-Ofast']
compilers = ['icc', 'clang', 'gcc']
results = []

for compiler in compilers:
    for opt in opt_flags:
        for m in range(0,M):
            N = Ns[m]
            logger.info('Running: make clean')
            clean_result = subprocess.run(["make", "clean"], capture_output=True, text=True)

            if clean_result.returncode != 0:
                logger.error(clean_result)
                raise Exception("Make clean failed.")

            logger.info(f'Running: make CC="{compiler}" CPPFLAGS="-DN={N}" CFLAGS="-march=native {opt}" TARGETS="tiny_md"')
            make_result = subprocess.run(["make", f'CC="{compiler}"', f'CPPFLAGS="-DN={N}"', f"CFLAGS=-march=native {opt}", 'TARGETS=tiny_md'], capture_output=True, text=True)

            if make_result.returncode != 0:
                logger.error(make_result)
                raise Exception("Make failed.")

            for _ in range(0,runs[m]):
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
                        results.append((particulas_s, N, opt, compiler, runs[m]))
                    logger.info(line)

with open(f'results/{timestamp}.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['particulas/s', 'N', 'opt', 'compiler', 'runs'])
    writer.writerows(results)
