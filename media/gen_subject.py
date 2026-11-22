import numpy as np, json, math
from PIL import Image
import scipy.ndimage as ndi

SRC = Image.open('/tmp/arena-coding-uploads/f6306d69-d76f-4224-a30f-516b89a34ff1.jpeg').convert('RGB')
A = np.asarray(SRC).astype(np.float32)/255.
h,w,_ = A.shape
lum = A @ np.array([0.299,0.587,0.114],np.float32)

def smoothstep(e0,e1,x):
    t=np.clip((x-e0)/(e1-e0),0,1); return t*t*(3-2*t)

# ---------- matte ----------
m0 = ndi.binary_fill_holes(lum > 0.022)
lbl,n = ndi.label(m0)
sizes = ndi.sum(m0,lbl,range(1,n+1)); sil = (lbl==int(np.argmax(sizes))+1)
sil = ndi.binary_fill_holes(ndi.binary_closing(sil,np.ones((9,9))))
sil[int(h*0.90):,:] = sil[int(h*0.90):,:] | ndi.binary_fill_holes(sil[int(h*0.90):,:])
ero  = ndi.binary_erosion(sil,np.ones((9,9)))
alpha = np.maximum(np.clip((lum-0.010)/0.055,0,1)*sil, ndi.gaussian_filter(ero.astype(np.float32),2.5))
alpha = np.clip(ndi.gaussian_filter(alpha,1.0),0,1)
alpha[int(h*0.93):,:] = np.maximum(alpha[int(h*0.93):,:], ndi.binary_erosion(sil[int(h*0.93):,:],np.ones((15,15))).astype(np.float32))

# ---------- grade ----------
c = A.copy()
s_mask = np.clip(1.0 - lum/0.30,0,1)**1.35
h_mask = smoothstep(0.40,1.0,lum)
c += s_mask[...,None]*np.array([-0.010,0.018,0.060],np.float32)
c += h_mask[...,None]*np.array([0.070,0.026,-0.042],np.float32)
c = np.clip((np.clip(c,0,1)-0.5)*1.06+0.5,0,1)
g = c.mean(2,keepdims=True); c = np.clip(g+(c-g)*1.05,0,1)
edge = np.zeros((h,w),np.float32); edge[:,1:]=np.abs(np.diff(alpha,axis=1))
edge = ndi.gaussian_filter(edge,3.0)
xxn = np.linspace(0,1,w)[None,:]
rimL = np.clip(edge*3.2,0,1)*np.clip(1-xxn*1.9,0,1)
rimR = np.clip(edge*3.2,0,1)*np.clip(xxn*1.7-0.6,0,1)
c += rimL[...,None]*np.array([0.22,0.36,0.55],np.float32)
c += rimR[...,None]*np.array([0.60,0.34,0.10],np.float32)
c = np.clip(c,0,1)

img = Image.fromarray((np.dstack([c,alpha])*255).astype(np.uint8),'RGBA')

# ---------- extend torso downward ----------
EXT = 1500
cw, ch = img.width, img.height
arr = np.asarray(img).astype(np.float32)/255.
tail = arr[ch-40:ch].mean(0)
xx = np.linspace(-1,1,cw)[None,:]
yc = np.linspace(0,1,EXT)[:,None]
body = np.tile(tail[None,:,:3],(EXT,1,1))
body *= (1.0-0.62*yc)[...,None]
body *= (1.0-0.42*np.clip(np.abs(xx)-0.50,0,1)/0.50)[...,None]
body_a = tail[None,:,3]*np.exp(-yc*EXT/620.0)*np.clip(1.0-0.9*np.clip(np.abs(xx)-0.72,0,1)/0.28,0,1)
out = np.zeros((ch+EXT,cw,4),np.float32); out[:ch]=arr
out[ch:]=np.dstack([np.clip(body,0,1),body_a])
k=40
for i in range(k):
    t=i/(k-1)
    out[ch-k+i,:,:3]=arr[ch-k+i,:,:3]*(1-t)+out[ch-k+i,:,:3]*0+t*np.clip(body[0],0,1)
    out[ch-k+i,:,3] =arr[ch-k+i,:,3]*(1-t)+t*body_a[0]
a_img = Image.fromarray((np.clip(out,0,1)*255).astype(np.uint8),'RGBA')

# ---------- defringe (colour bleed into transparent pixels) ----------
arr2 = np.asarray(a_img).astype(np.float32)/255.
rgb, al = arr2[...,:3], arr2[...,3]
col = rgb.copy(); mask = (al > 0.02)
col *= mask[...,None]; wt = mask.astype(np.float32)
for _ in range(14):
    colb = np.stack([ndi.uniform_filter(col[...,k],3) for k in range(3)],2)
    wb = ndi.uniform_filter(wt,3)
    newpx = (wb>0.02) & (~mask)
    col[newpx] = colb[newpx]/np.maximum(wb[newpx],1e-4)[:,None]
    wt[newpx]=1.0; mask = mask|newpx
rgb[al<=0.02] = col[al<=0.02]
al2 = np.clip(ndi.gaussian_filter(al,0.7),0,1)
final = Image.fromarray((np.clip(np.dstack([rgb,al2]),0,1)*255).astype(np.uint8),'RGBA')

# ---------- pose: rotate +21deg so the head is upright ----------
FACE=(789.0,1018.0); ANG=21.0
rot = final.rotate(ANG, resample=Image.BICUBIC, expand=True)
nw,nh = rot.size
cx,cy = cw/2.0, (ch+EXT)/2.0
th=math.radians(ANG)
dx,dy = FACE[0]-cx, FACE[1]-cy
vx,vy = dx,-dy
rx = vx*math.cos(th)-vy*math.sin(th); ry = vx*math.sin(th)+vy*math.cos(th)
fx = nw/2.0+rx; fy = nh/2.0-ry
rot.save('media/subject_rot.png')
json.dump({'face':[fx,fy],'size':[nw,nh],'angle':ANG}, open('media/subject_rot.json','w'))
print('subject ok', rot.size, fx, fy)
