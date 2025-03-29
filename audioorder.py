import numpy as np 
import matplotlib.pyplot as plt 
from scipy.io import wavfile
from scipy.signal import periodogram
import argparse
import sys

# starting at C#2 to D#6
pbkeys = np.array([7,10,12,14,17,19,22,24,26,29,31,34,36,38,41,43,46,48,50,53,55,58,60,62,65,67])

# from f(n) = 2**((n-49)/12)*440 (Hz)
pbfreqs = 2**((pbkeys-49.)/12.)*440

# list of first 200 primes 
primes = [
2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,
89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,
181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,
283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,
401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,
509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,
631,641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,
743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,
859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,
971,977,983,991,997,1009,1013,1019,1021,1031,1033,1039,1049,1051,1061,
1063,1069,1087,1091,1093,1097,1103,1109,1117,1123,1129,1151,1153,1163,
1171,1181,1187,1193,1201,1213,1217,1223
]

sample_rate=44100 # Hz
max_amp_default = 15000

ap = argparse.ArgumentParser()
ap.add_argument('--max_amp','-ma',help='Maximum digital amplitude of wave (16 bit audio should be less than 32768)',required=False,default=max_amp_default)
ap.add_argument('--mix','-m',help='Mix of base tone and modular exponents trace (1 = pure tone, 0 = modexp only, default = 0.8)',required=False,default=0.8)
ap.add_argument('--num_note_values','-nnv',help='Number of note values to sample (starting with whole note = 1 beat (default=4) reducing by 1/2^nnv',required=False,default=4)
ap.add_argument('--num_notes','-nn',help='Number of notes in half-sweep of tones (default=20)',required=False,default=20)
ap.add_argument('--integers_to_play','-itp',help='Comma-separated list of integers to play',required=False,default='')
ap.add_argument('--values_to_play','-vtp',help='Comma-separated list of note times (values, in seconds) to play',required=False,default='')
ap.add_argument('--number_of_sweeps','-ns',help='Number of soft/loud/soft sweeps (default=3)',required=False,default=3)
ap.add_argument('--start_integer_to_play','-sip',help='Closest prime number (greater than this value) to start playing',required=False,default=150)
ap.add_argument('--num_integers_to_play','-nip',help='Number of primes after start to use in generating notes',required=False,default=0)
ap.add_argument('--primes','-p',help='Choose primes in sequence (not sequential integers) default=False',required=False,default=False,action='store_true')
ap.add_argument('--prime_pairs','-pp',help='Choose from lowest several hundred or so prime pairs in sequence, default=False',required=False,default=False,action='store_true')
ap.add_argument('--include_squares','-ixs',help='Include ord(a)=2 terms, default=False',required=False,default=False,action='store_true')
ap.add_argument('--random_base','-rb',help='Pick random modular base, othewise use element with lowest order, default=False',required=False,default=False,action='store_true')
ap.add_argument('--wave_file','-wf',help='Wav file name to output (default=integers.wave)',required=False,default='integers.wav')
ap.add_argument('--generate_spectrogram','-gs',help='Generate spectrogram of resulting wav file, default=False',required=False,default=False,action='store_true')
ap.add_argument('--noreverse','-nrv',help='Do not reverse order of notes on downward sweep, default=False',required=False,default=False,action='store_true')
ap.add_argument('--spectrogram_file','-sf',help='Spectrogram (png) file name to output (default='')',required=False,default='')
ap.add_argument('--spectrogram_scale','-ss',help='Spectrogram scaling (log,lin,sqrt (default=sqrt)',required=False,default='sqrt')
ap.add_argument('--raw_freq','-rf',help='Use raw integer as frequency; otherwise assign to closest pentatonic note; default=False',required=False,default=False,action='store_true')
ap.add_argument('--beat','-bt',help='Beat interval (s) default=1 s',required=False,default=1)
ap.add_argument('--selected_trace_file','-stf',help='File name for plot of selected traces/power spectral densities (default='')',required=False,default='')
ap.add_argument('--num_selected_traces','-nst',help='Plot the first [nst] traces (default=0, i.e. no plot)',required=False,default=0)

clargs = ap.parse_args(sys.argv[1:])
num_note_values=int(clargs.num_note_values)
num_notes=int(clargs.num_notes)
max_amp = int(clargs.max_amp)
mix = float(clargs.mix)
number_of_sweeps = int(clargs.number_of_sweeps)
beat = float(clargs.beat)

integers_to_play=[]
# specific list on command line
if len(clargs.integers_to_play) > 0:
	integers_to_play=[int(x) for x in clargs.integers_to_play.split(',')]	
	num_notes=len(integers_to_play)

# generate list from other settings
nip = int(clargs.num_integers_to_play)
sip = int(clargs.start_integer_to_play)
if (clargs.primes or clargs.prime_pairs) and nip==0: # use primes
	print("Warning: No number of integers specified - using 10")
	nip=10
if nip > 0 and len(integers_to_play)==0:
	if clargs.primes or clargs.prime_pairs: # use primes
		if clargs.prime_pairs:
			trunc_primes = np.array(primes[4:24])
			op = np.unique(np.outer(trunc_primes,trunc_primes).reshape(len(trunc_primes)**2))
			# remove squares along diagonal
			diag = trunc_primes**2
			pr = np.array([x for x in op if not x in diag])
			pr.sort()
		else:
			pr = primes
		idx = np.argmin(np.abs(sip-np.array(pr)))
		if pr[idx] < sip:
			idx+=1
		integers_to_play=pr[idx:idx+nip]
	else: # sequential integers
		integers_to_play=np.arange(sip,sip+nip)
	num_notes=len(integers_to_play)

if len(clargs.values_to_play) > 0:
	note_durations=[]
	for tok in clargs.values_to_play.split(','):
		note_durations.append(float(tok))
else:
	note_durations=np.array([1./2**x for x in range(num_note_values)])

# if we specified note values on command line
if len(clargs.values_to_play)>0 and len(note_durations)==num_notes:
	note_times = beat*np.array(note_durations)
else: # otherwise randomly sample note values
	note_times=beat*np.array([note_durations[int(i*num_note_values)] for i in np.random.rand(num_notes)])

note_spacing=-0.02 # s, negative overlaps ok (preferred)

def fix_wave_edges(wave,edge=0.02):
	# fix up edges for gradual on/off, remove DC component and normalize
	nwave = wave-np.mean(wave)
	max_wave = max(nwave)
	min_wave= min(nwave)
	if max_wave-min_wave==0:
		norm=1
	else:
		norm=max_wave-min_wave
	nwave = 2*(nwave-min_wave)/norm - 1
		
	t_env = np.arange(0,edge,1/sample_rate)
	env_l = t_env/edge # 0-1 in edge sec
	nwave[:len(t_env)]=nwave[:len(t_env)]*env_l
	nwave[len(nwave)-len(t_env):]=nwave[len(nwave)-len(t_env):]*(1-env_l)
	return nwave

def gen_waves(note_freqs=[]):
	waves=[]
	if len(note_freqs)==0:
		note_freqs = np.array([pbfreqs[int(i*len(pbfreqs))] for i in np.random.rand(num_notes)])
	for t,f in zip(note_times,note_freqs):
		wave=np.sin(2*np.pi*f*np.arange(0,t,1/sample_rate))
		waves.append(fix_wave_edges(wave))
	return(waves)


def compute_order(p):
	# brute force order finding
	candidates=[]
	coprimes=[]
	charmichael=[]
	for a in range(2,p):
		if np.gcd(a,p)==1:
			orders=[]
			smallest=False
			for x in range(1,p):
				if pow(a,x,int(p))==1:
					orders.append(x)
					if not smallest:
						candidates.append(x)
						coprimes.append(a)
						smallest=True
			charmichael.append(orders)

	# find Charmichael function
	# lowest common exponent in list of all coprimes
	charmichael_val = 1 
	if len(charmichael) > 0:
		cmf = []
		for x in charmichael[0]:
			inall=True
			for y in charmichael:
				if not x in y:
					inall=False
					break
			if inall:
				cmf.append(x)
		if len(cmf) > 0:
			charmichael_val = min(cmf)

	# Find coprime that gives minimum order
	# The minimum order is almost always '2' because
	# (p-1)^2 = 1 mod p  
	# because p^2 - 2p + 1 = np + 1 means n = p-2 always works
	# Optionally exclude these squares
	if len(candidates)>0:
		idx=0
		if not clargs.include_squares:
			npcandidates = np.array(candidates)
			filt = npcandidates > 2
			if len(npcandidates[filt]) > 0:
				if not clargs.random_base:
					idx = np.argmin(npcandidates[filt])
				else:
					idx = int(len(npcandidates[filt])*np.random.rand())
				r=npcandidates[filt][idx]
				a=np.array(coprimes)[filt][idx]
			else: # all are order 2
				a=coprimes[0]
				r=candidates[0]
		else:
			if clargs.random_base:
				idx = np.argmin(candidates) 
			else:
				idx = int(len(candidates)*np.random.rand())
			r=candidates[idx]
			a=coprimes[idx]
	else:
		a=p
		r=1
	return(r,a,charmichael_val)

def gen_integer_waves(sampled_integers=[]):
	waves=[]
	tones=[]
	# default: random sampling of primes
	if len(sampled_integers)!=num_notes:
		sampled_integers=np.array([primes[int(i*len(primes))] for i in np.random.rand(num_notes)])
	# find an example coprime
	sampled_coprimes=[]
	sampled_orders=[]
	charmichael=[]
	for p in sampled_integers:
		r,a,c = compute_order(p)
		sampled_coprimes.append(a)
		sampled_orders.append(r)
		charmichael.append(c)

	nst = int(clargs.num_selected_traces)
	if nst > 0:
		plot_traces(sampled_integers[:nst], sampled_orders[:nst],sampled_coprimes[:nst])

	# now compute modular exponents a^r mod p as wave
	print("#val(s)\tN\tf(Hz)\ta\tr\tl(N)\tF")
	for a,r,p,t,c in zip(sampled_coprimes,sampled_orders,sampled_integers,note_times,charmichael):
		# mod exp of primes with first coprime
		mes = [pow(int(a),int(x),int(p)) for x in range(p)]
		# each mod sample time is at least two audio samples
		t_p_minsam = 2/sample_rate
		min_tone = 1/(p*t_p_minsam)
		if clargs.raw_freq:
			tone = min_tone
		else:
			# closest black-key tone to prime
			tone_idx = np.argmin(np.abs(min_tone-pbfreqs))
			tone = pbfreqs[tone_idx]
		tones.append(tone)
		t_p_sam = 1/(tone*p)
		# wave time base
		t_base=np.arange(0,t,1/sample_rate)
		factor = np.gcd(pow(int(a),int(r/2),int(p))-1,p)
		print("{0:.2f}\t{1:04d}\t{2:.2f}\t{3:04d}\t{4}\t{5}\t{6}".format(t,p,tone,a,r,c,factor))
		# approximate number of repeats of this prime mapped to tone in t
		reps = int(np.ceil(t/(t_p_sam*p)))
		mes_reps = reps*mes
		t_mes_reps = np.arange(0,len(mes_reps))*t_p_sam
		wave = np.interp(t_base,t_mes_reps,mes_reps)
		# now resample onto note 
		waves.append(fix_wave_edges(wave))

	return(waves,tones,sampled_integers)

def gen_sweep(waves):
	t_half = sum(note_times)+(len(note_times)-1)*note_spacing
	half_sweep=np.zeros(int(np.ceil(sample_rate*t_half)))
	c_time=0
	for t,wave in zip(note_times,waves):
		ti=int((c_time+note_spacing)*sample_rate)
		if ti<0:
			ti=0
		wend=int(ti+len(wave))
		trim=int(len(wave))
		if wend-ti!=trim:
			trim=wend-ti
		half_sweep[ti:wend]+=wave[:trim]
		c_time+=(t+note_spacing)
	
	# remove wave edges at end
	n_end_trim = int(np.ceil(np.abs(sample_rate*note_spacing)))
	if len(half_sweep) > n_end_trim:
		half_sweep = half_sweep[:-n_end_trim]

	N_half=len(half_sweep)
	idx_half_sweep=np.arange(0,N_half)
	t_half_sweep=idx_half_sweep/sample_rate
	amp=max_amp*idx_half_sweep/(N_half-1) # modulate signal by this much

	# align min, max to closest sample freq
	if clargs.noreverse:
		full_sweep=np.append(amp*np.array(half_sweep),amp[::-1]*np.array(half_sweep))
	else:
		full_sweep=np.append(amp*np.array(half_sweep),-amp[::-1]*np.array(half_sweep[::-1]))
	full_amp = np.append(amp,amp[::-1])
	return full_sweep,full_amp

def gen_repeats(full_sweep,full_amp):
	# lists work better here
	repeat_sweeps=[]
	for sn in range(number_of_sweeps):
		repeat_sweeps.extend(list(full_sweep))
	return repeat_sweeps
	
def gen_spectrogram(repeat_sweeps):
	# create psd spectrogram 
	fft_int = 1024 # around 50 ms
	pgs = []
	f_min=20
	f = sample_rate*np.linspace(0,0.5,int(fft_int/2))
	f[0]=1e-20 # avoid bad log val
	# the array index corresponding to a given freq on log10 scale
	f_LUT = np.array((np.log10(f/f_min)/np.log10(f[-1]/f_min))*len(f))
	f_LUT[f_LUT<0]=0 # get rid of freq less than f_min
	f_LUT_idx = np.arange(0,len(f_LUT))
	pg_idx = np.arange(0,fft_int/2+1)
	
	for tc in range(0,len(repeat_sweeps),fft_int):
		pf, pg = periodogram(repeat_sweeps[tc:tc+fft_int],nfft=fft_int,window='hamming')
		# map onto log10 scale for f
		map_pg = np.zeros(len(pg))
		for i in range(len(pg)-1):
			idx = np.interp(i, f_LUT, f_LUT_idx)
			map_pg[i]=np.interp(idx,pg_idx, pg)
		pgs.append(map_pg)
	
	# transform data here
	if clargs.spectrogram_scale=='log':
		pgim = np.log(np.array(pgs).transpose())
	elif 'lin' in clargs.spectrogram_scale:
		pgim = np.array(pgs).transpose()
	else: # sqrt
		pgim = np.sqrt(np.array(pgs)).transpose()
	
	ext=[0,len(repeat_sweeps)/sample_rate,np.log10(f_min),np.log10(sample_rate*pf[-1])]
	plt.imshow(pgim,extent=ext,aspect='auto',origin='lower') # cmap='nipy_spectral'
	plt.xlabel("Time (s)")
	plt.ylabel("log$_{10}(f)$ (Hz)")
	if len(clargs.spectrogram_file) > 0:
		plt.savefig(clargs.spectrogram_file)
		plt.clf()
	else:
		plt.show()

def plot_traces(integers,orders,coprimes):
	""" Plot time and frequency traces of for selected integers
	"""
	
	fig, axs = plt.subplots(ncols=2,nrows=len(integers))
	maxint=max(integers)
	for ax,i,o,cp in zip(axs,integers,orders,coprimes):
		xs = range(1,maxint)
		mes = [pow(int(cp),int(x),int(i)) for x in xs]
		ax[0].plot(xs,mes,label=r"$%s^{%s}$ mod %s"%(str(cp),str(o),str(i)))
		ax[0].legend()
		f,Pxx = periodogram(mes,window='hamming')
		ax[1].plot(1e-3*sample_rate*f,Pxx)
		ax[0].set_xlim(0,max(integers))
		# turn off xticklabels for all but last
		if i != integers[-1]:
			ax[0].set_xticklabels([])
			ax[1].set_xticklabels([])
		else:
			ax[0].set_xlabel("$x < p$")
			ax[1].set_xlabel("f (kHz)")
		ax[1].set_yticklabels([])
	
	plt.subplots_adjust(hspace=0.025,wspace=0.05)
	if len(clargs.selected_trace_file) > 0:
		plt.savefig(clargs.selected_trace_file)
		plt.clf()
	else: # plot to interactive
		plt.show()
	
	
########
# Start 
########

waves, tones, ints  = gen_integer_waves(integers_to_play)
pure_waves = gen_waves(tones)

# now build sweep from note waves
mod_full_sweep,mod_full_amp = gen_sweep(waves)
pure_full_sweep,pure_full_amp = gen_sweep(pure_waves)

full_sweep = (1-mix)*mod_full_sweep + mix*pure_full_sweep
full_amp = (1-mix)*mod_full_amp + mix*pure_full_amp

# make repeats
repeat_sweeps = gen_repeats(full_sweep, full_amp)

# write to file
wavfile.write(clargs.wave_file,int(sample_rate),np.array(repeat_sweeps,dtype=np.int16))

# generate spectrogram
if clargs.generate_spectrogram:
	gen_spectrogram(repeat_sweeps)
