import sys, time
sys.path.insert(0, 'media')
from PIL import Image
import scene as S

N = 270
miss = [int(l) for l in open('media/missing.txt').read().split() if l.strip()]
t0 = time.time()
for k, i in enumerate(miss):
    Image.fromarray(S.post(S.render(i, N), i, N)).save(f'media/frames/f_{i:04d}.png', compress_level=1)
    print(f'{k+1}/{len(miss)} {i} {time.time()-t0:.0f}s', flush=True)
print('done', round(time.time()-t0, 1), flush=True)
