import sys, os, time
sys.path.insert(0, 'media')
import numpy as np
from PIL import Image
import multiprocessing as mp
import scene as S

FPS = 30
DUR = 9.0
N = int(round(FPS*DUR))
OUT = 'media/frames'


def work(i):
    try:
        a = S.post(S.render(i, N), i, N)
        Image.fromarray(a).save(f'{OUT}/f_{i:04d}.png', compress_level=1)
        return i
    except Exception as ex:
        return ('err', i, repr(ex))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    done = 0
    with mp.Pool(2) as pool:
        for r in pool.imap_unordered(work, range(N), chunksize=2):
            done += 1
            if done % 10 == 0 or done == N:
                el = time.time()-t0
                print(f'{done}/{N}  {el:.0f}s  eta {el/done*(N-done):.0f}s', flush=True)
            if isinstance(r, tuple) and r[0] == 'err':
                print('ERR', r[1], r[2], flush=True)
    print('done in', round(time.time()-t0, 1), 's', flush=True)
