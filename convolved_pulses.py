import numpy as np 
import matplotlib.pyplot as plt

def convolved_pulse(omega, tl, te, ld, ll):
	return  (te*tl*np.exp(-tl*abs(omega))*abs(tl*omega)**(-1j*2*ll/np.pi*tl*omega))/((1-1j*ld*te*omega)*(1+1j*(1-ld) *te*omega))

Xlen=2**17
dX = 1
Xfreq = np.fft.rfftfreq(Xlen, dX)
u = 2*np.pi*Xfreq
X = np.arange(-Xlen/2+1,Xlen/2+1)*dX

C = convolved_pulse(u, 19, 180, 0.8, 0.5)
spectra = C * np.conj(C)

waveform = np.fft.irfft(C[:])[::-1]/dX
waveform = np.fft.fftshift(waveform)

plt.figure()
plt.loglog(Xfreq, spectra)

plt.figure()
plt.plot(X,waveform)
plt.xlim(-1000,1000)

plt.show()
