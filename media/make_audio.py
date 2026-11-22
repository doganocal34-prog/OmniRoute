import numpy as np, wave
SR=48000; DUR=9.0
n=int(SR*DUR); t=np.arange(n)/SR
rng=np.random.default_rng(11)

def bp(x, lo, hi):
    X=np.fft.rfft(x); f=np.fft.rfftfreq(len(x),1/SR)
    Y=X*np.exp(-((np.log2(np.maximum(f,1)/np.sqrt(lo*hi)))**2)*2.2)
    return np.fft.irfft(Y, len(x))

def brown(n):
    w=rng.normal(0,1,n); out=np.cumsum(w); out-=np.linspace(out[0],out[-1],n)
    return out/np.max(np.abs(out))

wind=bp(brown(n),60,900); wind/=np.max(np.abs(wind))+1e-9
gust=0.55+0.45*np.sin(2*np.pi*0.13*t+0.4)+0.18*np.sin(2*np.pi*0.31*t+1.9)
L=wind*gust
rum=bp(brown(n),28,130); rum/=np.max(np.abs(rum))+1e-9
L+=0.42*rum
hiss=bp(rng.normal(0,1,n),2600,11000); hiss/=np.max(np.abs(hiss))+1e-9
rain=0.10*(0.55+0.45*np.sin(2*np.pi*0.07*t+2.2))
L+=hiss*rain
sub=np.sin(2*np.pi*42*t)*0.16+np.sin(2*np.pi*63*t)*0.07
env=np.clip(t/1.2,0,1)*np.clip((DUR-t)/2.2,0,1)**0.8
L+=sub*env
R=bp(brown(n),60,900); R/=np.max(np.abs(R))+1e-9
R=R*(0.55+0.45*np.sin(2*np.pi*0.11*t+2.6))+0.42*rum+hiss*rain+sub*env
pan=0.5+0.5*np.sin(2*np.pi*0.06*t)
left=L*(0.82+0.18*(1-pan)); right=R*(0.82+0.18*pan)
fade=np.clip(t/0.8,0,1)*np.clip((DUR-t)/1.1,0,1)
left*=fade; right*=fade
st=np.stack([left,right],1); st=st/(np.max(np.abs(st))+1e-9)*0.72
st=np.tanh(st*1.25)/1.25; st=st/np.max(np.abs(st))*0.78
pcm=(st*32767*0.9).astype('<i2')
with wave.open('media/ambience.wav','wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print('audio ok', pcm.shape)
