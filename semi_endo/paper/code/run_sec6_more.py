# Remaining news horizons for the slide deck (monthly calibration), same output format as run_sec6.py.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_sec6
if __name__ == '__main__':
    from multiprocessing import Pool
    jobs = [(ttb, nm, K, 'news') for K in (4, 6, 7, 8, 9, 10, 11, 12) for nm in ('TFP shock', 'demand shock') for ttb in (True, False)]
    jobs.sort(key=lambda j: (j[0], j[1] == 'demand shock', -j[2]))     # slow no-TTB TFP jobs first
    with Pool(14) as p: p.map(run_sec6.run, jobs, chunksize=1)
    print('MORE ALL DONE', flush=True)
