"""2.5D cinematic drone pull-back compositor (1080x1920, 9:16)."""
import numpy as np, math, json
from PIL import Image, ImageFilter

W, H = 1080, 1920
A = 'media'

SUBJ = Image.open(f'{A}/subject_rot.png').convert('RGBA')
FACE = tuple(json.load(open(f'{A}/subject_rot.json'))['face'])

# ---- environment plate: photoreal night rooftop ----
env0 = Image.open(f'{A}/plate_aerial.png').convert('RGB')
ENV = env0.resize((int(env0.width*2.1), int(env0.height*2.1)), Image.LANCZOS)
ENV = ENV.filter(ImageFilter.UnsharpMask(radius=2.0, percent=55, threshold=2))
a = np.asarray(ENV, np.float32)/255.
lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
sh = np.clip(1.0-lum/0.22, 0, 1)[..., None]**1.15
a = a + sh*np.array([-0.012, 0.010, 0.030], np.float32)
a = np.clip((a-0.5)*1.07+0.47, 0, 1)
a = np.clip(a*0.92, 0, 1)
ENV = Image.fromarray((a*255).astype(np.uint8))
ENV_L = ENV.filter(ImageFilter.GaussianBlur(22))


def smootherstep(x):
    x = np.clip(x, 0, 1)
    return x*x*x*(x*(x*6-15)+10)


def lerp(a, b, t):
    return a+(b-a)*t


def expo(a, b, t):
    return a*(b/a)**t


_cache = {}


def scaled(img, s):
    key = (id(img), round(float(s), 4))
    if key in _cache:
        return _cache[key]
    nw, nh = max(1, int(round(img.width*s))), max(1, int(round(img.height*s)))
    out = img.resize((nw, nh), Image.LANCZOS if s < 1 else Image.BICUBIC)
    if len(_cache) > 80:
        _cache.clear()
    _cache[key] = out
    return out


def place(img, s, anchor_src, anchor_dst, rot=0.0, out=(W, H)):
    s = max(float(s), 1e-4)
    im = scaled(img, s)
    th = math.radians(rot)
    c, sn = math.cos(th), math.sin(th)
    axs, ays = anchor_src[0]*s, anchor_src[1]*s
    axd, ayd = anchor_dst
    return im.transform(out, Image.AFFINE,
                        (c, sn, axs-(c*axd+sn*ayd), -sn, c, ays-(-sn*axd+c*ayd)),
                        resample=Image.BICUBIC)


def stamp(img, cx, cy, r, col, inten, sharp=2.0):
    if r <= 0.4 or inten <= 0.0004:
        return
    h, w, _ = img.shape
    rr = int(r*3.0)+1
    x0, x1 = int(cx)-rr, int(cx)+rr
    y0, y1 = int(cy)-rr, int(cy)+rr
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return
    xa, xb = max(0, x0), min(w, x1)
    ya, yb = max(0, y0), min(h, y1)
    ys, xs = np.mgrid[ya:yb, xa:xb]
    d = np.sqrt((xs-cx)**2+(ys-cy)**2)/r
    g = np.clip(1.0-d, 0, 1)**sharp
    img[ya:yb, xa:xb] += g[..., None]*np.array(col, np.float32)*inten


YY = np.arange(H, dtype=np.float32)[:, None].repeat(W, 1)
XX = np.arange(W, dtype=np.float32)[None, :].repeat(H, 0)
NX, NY = XX/W, YY/H

# ---- parallax light layers ----
rng = np.random.default_rng(777)
BOK = []
for band, (cnt, ylo, yhi, rlo, rhi, ilo, ihi, par) in enumerate([
        (60, 0.30, 0.62, 0.004, 0.018, 0.10, 0.30, 0.05),
        (40, 0.45, 0.70, 0.008, 0.030, 0.10, 0.26, 0.14)]):
    for _ in range(cnt):
        c = (1.0, 0.62, 0.28) if rng.random() < 0.7 else (0.45, 0.70, 1.0)
        BOK.append(dict(x=rng.uniform(-0.1, 1.1), y=rng.uniform(ylo, yhi),
                        r=rng.uniform(rlo, rhi), col=c, i=rng.uniform(ilo, ihi),
                        par=par, tw=rng.uniform(0, 6.28)))

MOTES = [dict(x=rng.uniform(-0.08, 1.08), y=rng.uniform(-0.02, 1.02),
              r=rng.uniform(0.0025, 0.010), par=rng.uniform(1.1, 2.0),
              i=rng.uniform(0.3, 1.0), ph=rng.uniform(0, 6.28)) for _ in range(30)]

RAIN = [dict(x=rng.uniform(-0.05, 1.05), y=rng.uniform(0, 1.0), w=rng.uniform(0.6, 1.9),
             l=rng.uniform(0.022, 0.075), v=rng.uniform(1.5, 3.4),
             par=rng.uniform(0.35, 1.0), a=rng.uniform(0.22, 0.75)) for _ in range(150)]

E_ANCHOR = (ENV.width*0.500, ENV.height*0.545)   # railing line, centre
ENV_SIZE = ENV.size


def add_rain(img, u, e, SX, SY):
    h, w, _ = img.shape
    for R in RAIN:
        yy = (R['y'] + u*R['v']*1.9) % 1.25 - 0.12
        xx = (R['x'] + u*0.045*R['par'] + SX/1000.0*R['par']*1.1) % 1.1 - 0.05
        py = yy + SY/1000.0*R['par']*1.1
        y0, y1 = int(py*h), int(py*h + R['l']*h*1.5)
        x0, x1 = int(xx*w), int(xx*w + R['w']*2)
        if x1 < 0 or x0 >= w or y1 < 0 or y0 >= h:
            continue
        xa, xb = max(0, x0), min(w, x1)
        ya, yb = max(0, y0), min(h, y1)
        if xb <= xa or yb <= ya:
            continue
        seg = img[ya:yb, xa:xb]
        fade = np.linspace(1.0, 0.25, yb-ya)[:, None, None]
        seg += fade*np.array([0.55, 0.62, 0.70], np.float32)*R['a']*0.30*(1.0-0.35*e)
    return img


def camera(u):
    e = smootherstep(np.clip((u-0.01)/0.95, 0, 1))
    e = 0.95*e + 0.05*u
    sx = 26.0*math.sin(u*math.pi*0.9+0.25) - 8.0*math.sin(u*math.pi*2.2)
    sy = -18.0*e + 9.0*math.sin(u*math.pi*0.75+0.6)
    rot = -0.55 + 0.95*e + 0.12*math.sin(u*math.pi*1.6)
    vib = 1.4*math.exp(-u*3.6)
    return e, sx+vib*math.sin(u*54.0), sy+vib*math.cos(u*62.0+0.4), rot


def pblur(x, r):
    if r <= 0.05:
        return x
    return np.asarray(Image.fromarray((np.clip(x, 0, 1)*255).astype(np.uint8))
                      .filter(ImageFilter.GaussianBlur(r)), np.float32)/255.


def render(i, n=270):
    u = i/max(n-1, 1)
    e, SX, SY, ROT = camera(u)
    rg = np.random.default_rng(3100+i)
    SX += rg.normal(0, 0.45)
    SY += rg.normal(0, 0.45)

    # ---------- environment ----------
    es = expo(2.55, 0.995, e)
    edx = lerp(0.315, 0.500, e) + SX/1000.0*0.55
    edy = lerp(0.330, 0.535, e) + SY/1000.0*0.55
    env = place(ENV, es, E_ANCHOR, (edx*W, edy*H), ROT*0.55)
    comp = np.asarray(env, np.float32)/255.
    lift = np.clip((e-0.25)/0.75, 0, 1)**1.4
    comp = comp*(1.0+0.26*lift) + (0.016*lift)*np.array([1.0, 0.74, 0.48], np.float32)

    # practical halo (upper-left): motivates the key light on his face
    stamp(comp, (0.075+SX/1000.0*0.5)*W, (0.115+SY/1000.0*0.5)*H, 0.26*W, (1.0, 0.66, 0.34), 0.16, 2.2)
    stamp(comp, (0.075+SX/1000.0*0.5)*W, (0.115+SY/1000.0*0.5)*H, 0.055*W, (1.0, 0.86, 0.66), 0.5, 2.0)

    # distant city bokeh
    for B in BOK:
        px = B['x']+SX/1000.0*B['par']*3.0
        py = B['y']+SY/1000.0*B['par']*3.0
        tw = 0.85+0.15*math.sin(u*6.0+B['tw'])
        stamp(comp, px*W, py*H, max(B['r']*W*(1+0.4*e), 1.1), B['col'],
              B['i']*tw*(1-0.35*e), 1.9)

    # ---------- the man ----------
    ss = expo(0.705, 0.268, e)
    sdx = lerp(0.552, 0.500, e) + SX/1000.0*1.05
    sdy = lerp(0.338, 0.238, e) + SY/1000.0*1.05
    subj = place(SUBJ, ss, FACE, (sdx*W, sdy*H), ROT*0.60)
    sp = np.asarray(subj.convert('RGBa'), np.float32)/255.
    al = sp[..., 3]

    # light wrap from the environment onto the silhouette
    lw = np.clip(pblur(al, 4.5)*(1-al)*4.2, 0, 1)
    envl = np.asarray(place(ENV_L, es, E_ANCHOR, (edx*W, edy*H), ROT*0.55), np.float32)/255.
    rgb = sp[..., :3]*(1-lw[..., None]*0.40)+envl*lw[..., None]*0.40

    # rims: warm from the lamp side, cool from the city
    ga = pblur(al, 1.5)
    band = np.clip((ga-pblur(al, 7.0))*5.0, 0, 1)
    warm = np.clip(1.0-(NX-0.02)/0.40, 0, 1)**1.5
    cool = np.clip((NX-0.34)/0.66, 0, 1)**1.5
    rgb = np.clip(rgb+band[..., None]*(warm[..., None]*np.array([0.34, 0.19, 0.06], np.float32)
                                      + cool[..., None]*np.array([0.10, 0.18, 0.28], np.float32)), 0, 1)

    # gentle warm falloff across the face + filmic shoulder
    face_light = 1.0+0.16*np.clip(1.0-(NX-0.05)/0.55, 0, 1)**1.6
    rgb = rgb*face_light[..., None]
    knee = 0.72
    over = np.clip(rgb-knee, 0, None)
    rgb = np.clip(np.clip(rgb, 0, knee)+over/(1.0+over*2.6), 0, 1)

    comp = rgb + comp*(1-al[..., None])

    # depth cue: soft dark pocket behind him + warm halo
    pocket = pblur(al, 70.0)*(1-al)*0.55*e
    comp = comp*(1.0-pocket[..., None]*0.55)
    halo = pblur(al, 24.0)*(1-al*0.92)
    comp = comp + halo[..., None]*np.array([0.048, 0.030, 0.016], np.float32)

    # near dust motes (strong parallax)
    for M in MOTES:
        px = M['x']+math.sin(u*math.pi*1.2+M['ph'])*0.04+SX/1000.0*M['par']
        py = M['y']-0.06*u*M['par']+math.cos(u*math.pi*0.85+M['ph'])*0.03+SY/1000.0*M['par']
        stamp(comp, px*W, py*H, M['r']*W*(1.7+1.7*e), (1.0, 0.80, 0.55),
              0.20*M['i']*(0.4+0.6*e), 2.4)

    # foreground bokeh near the lens
    for k, (bx, by, br, ii) in enumerate([(0.045, 0.985, 0.16, 0.10),
                                          (0.965, 0.90, 0.13, 0.07),
                                          (0.90, 1.02, 0.20, 0.06)]):
        px = bx+math.sin(u*2.0+k)*0.02+SX/1000.0*1.9
        py = by+math.cos(u*1.7+k)*0.02+SY/1000.0*1.9
        stamp(comp, px*W, py*H, br*W*(1+0.9*e), (1.0, 0.58, 0.26), ii*(0.5+0.5*e), 2.2)

    comp = add_rain(comp, u, e, SX, SY)

    # drifting fog in front
    off = int(SX*1.7+u*150)
    ofy = int(SY*1.7+u*70)
    fog = np.roll(np.roll(comp, ofy, 0), off, 1)
    fz = np.clip((pblur(fog.mean(2), 30.0)-0.035)*6.0, 0, 1)
    a2 = (fz*0.16*(0.6+0.4*math.sin(u*math.pi*0.9)))[..., None]
    comp = comp*(1-a2)+a2*np.array([0.055, 0.060, 0.068], np.float32)
    return comp


rr = np.sqrt(((NX-0.5)/0.62)**2+((NY-0.5)/0.60)**2)
VIG = np.clip(1.0-0.50*np.clip(rr-0.50, 0, 3)**1.7, 0.12, 1.0)
VIGN = np.stack([VIG*(1-0.10*np.clip(rr-0.6, 0, 2)), VIG,
                 VIG*(1-0.12*np.clip(rr-0.6, 0, 2))], 2).astype(np.float32)


def post(comp, i, n=270):
    u = i/max(n-1, 1)
    img = np.clip(comp, 0, 1)
    small = Image.fromarray((np.clip(img-0.60, 0, 1)*255).astype(np.uint8)).resize((W//4, H//4), Image.BILINEAR)
    b1 = np.asarray(small.filter(ImageFilter.GaussianBlur(2.6)).resize((W, H), Image.BILINEAR), np.float32)/255.
    b2 = np.asarray(small.filter(ImageFilter.GaussianBlur(10.0)).resize((W, H), Image.BILINEAR), np.float32)/255.
    bl = 0.85+0.55*u
    img = np.clip(img+b1*np.array([0.34, 0.27, 0.17], np.float32)*bl
                  + b2*np.array([0.20, 0.155, 0.11], np.float32)*bl, 0, 1)

    lum = img @ np.array([0.299, 0.587, 0.114], np.float32)
    sh = np.clip(1.0-lum/0.18, 0, 1)[..., None]**1.2
    hi = np.clip((lum-0.62)/0.38, 0, 1)[..., None]
    img = img+sh*np.array([-0.008, 0.014, 0.040], np.float32)+hi*np.array([0.022, 0.008, -0.014], np.float32)
    img = np.clip(img, 0, 1)
    img = np.clip((img-0.5)*1.05+0.49, 0, 1)
    dk = np.clip(1.0-lum/0.10, 0, 1)[..., None]
    g = img.mean(2, keepdims=True)
    img = img*(1-dk*0.5)+g*dk*0.5
    img = img*(1.0+0.10*u)

    ca = np.clip(rr-0.45, 0, 2)*0.7
    rch = np.roll(img[..., 0], 1, 1)*(1-ca)+img[..., 0]*ca
    bch = np.roll(img[..., 2], -1, 1)*(1-ca)+img[..., 2]*ca
    img = np.stack([rch, img[..., 1], bch], 2)

    fin = np.clip(u/0.05, 0, 1)**1.15
    fout = np.clip((1.0-u)/0.06, 0, 1)
    img = img*(fin*fout)*VIGN*1.14
    img = img+np.random.default_rng(4200+i).normal(0, 1, (H, W, 1)).astype(np.float32)*0.015
    return (np.clip(img, 0, 1)*255).astype(np.uint8)
