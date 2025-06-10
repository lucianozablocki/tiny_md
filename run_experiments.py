#!/usr/bin/env python3
import subprocess
import csv
import datetime
import os
import logging
from subprocess import call 

call("./opt/AMD/aocc-compiler-4.1.0/setenv_AOCC.sh", shell=True)
call("./opt/intel/oneapi/setvars.sh", shell=True)


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

M = 9
Ns = []
for m in range(1,M+1):
    Ns.append(4*m**3)

runs = [30,25,20,15,10,5,1,1,1]
opt_flags = ['-O1 -ffast-math -ftree-vectorize -funroll-loops']
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

            logger.info(f'Running: make CC="{compiler}" CPPFLAGS="-DN={N} -DSEED=0" CFLAGS="-march=native {opt}" TARGETS="tiny_md"')
            make_result = subprocess.run(["make", f'CC="{compiler}"', f'CPPFLAGS="-DN={N}"', f"CFLAGS=-march=native {opt}", 'TARGETS=tiny_md'], capture_output=True, text=True)

            if make_result.returncode != 0:
                logger.error(make_result)
                raise Exception("Make failed.")

            for _ in range(0,runs[m]):
                result = subprocess.run("perf stat -e fp_ret_sse_avx_ops.all,cpu-cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,dTLB-loads,dTLB-load-misses,l3_comb_clstr_state.request_miss ./tiny_md",
                   shell=True,
                   capture_output=True,
                   text=True)
                output = result.stdout
                output2 = result.stderr
                particulas_s = ''
                gflops = ''
                time = ''
                l1_misses = ''
                dtlb_misses = ''
                for line in output.split("\n"):
                    logger.info(line)
                    if '***' in line:
                        particulas_s = f'{
                            float(
                                line.split('***')[1] # sacar asteriscos
                                .replace('particulas/s:', "") # sacar el string "particulas/s:"
                                .strip() # remover cualquier leading/trailing whitespace
                                ):.3f
                            }'
                for line in output2.split("\n"):
                    logger.info(line)
                    if 'fp_ret_sse_avx_ops.all' in line:
                        gflops = f'{
                            int(
                                line.split('fp_ret_sse_avx_ops.all')[0]
                                .replace(",","")
                                .strip() # remover cualquier leading/trailing whitespace
                                )
                            }'
                    if 'L1-dcache-load-misses' in line and '<not counted>' not in line:
                        l1_misses = f'{
                            int(
                                line.split('L1-dcache-load-misses')[0]
                                .replace(",","")
                                .strip() # remover cualquier leading/trailing whitespace
                                )
                            }'
                    if 'dTLB-load-misses' in line and '<not counted>' not in line:
                        dtlb_misses = f'{
                            int(
                                line.split('dTLB-load-misses')[0]
                                .replace(",","")
                                .strip() # remover cualquier leading/trailing whitespace
                                )
                            }'
                    if 'seconds time elapsed' in line:
                        time = f'{
                            float(
                                line.replace('seconds time elapsed', "")
                                .strip() # remover cualquier leading/trailing whitespace
                                ):.3f
                            }'
                results.append((particulas_s, N, opt, compiler, runs[m], gflops, time, l1_misses, dtlb_misses))

with open(f'results/{timestamp}.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['particulas/s', 'N', 'opt', 'compiler', 'runs', 'GFLOPS', 'time', 'l1 misses', 'dtlb misses'])
    writer.writerows(results)
