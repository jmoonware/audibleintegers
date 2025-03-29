# to see all the command line options:
# python audioorder.py -h
# I set --mix 0.5 so the spectrograms are more visible
# audibly, the default 0.8 is less jarring sometimes

# defaults
python audioorder.py 

# integers in order from 200-230, generating spectrogram and 
# plotting first 10 traces
python audioorder.py -sip 200 -nip 30 -gs -sf spec200.png -nst 10 -stf traces200.png -wf integers200.wav --mix 0.5

# integers in order from 200-230, generating spectrogram and 
# plotting first 10 traces, using random 'a' base
python audioorder.py -sip 200 -nip 30 -gs -sf spec200rb.png -nst 10 -stf traces200rb.png -wf integers200rb.wav --mix 0.5 -rb

# integers in order from 200-230, generating spectrogram and 
# plotting first 10 traces, using random 'a' base, raw frequencies
python audioorder.py -sip 200 -nip 30 -gs -sf spec200rbrf.png -nst 10 -stf traces200rbrf.png -wf integers200rbrf.wav --mix 0.5 -rb -rf

#  30 primes in order from ~200, generating spectrogram and 
#  plotting first 10 traces, using random 'a' base
python audioorder.py -sip 200 -nip 30 -gs -sf spec200prb.png -nst 10 -stf traces200prb.png -wf integers200prb.wav --mix 0.5 -rb

#  30 prime pairs in order from ~200, generating spectrogram and 
#  plotting first 10 traces, using random 'a' base
python audioorder.py -sip 200 -nip 30 -gs -sf spec200pprb.png -nst 10 -stf traces200pprb.png -wf integers200pprb.wav --mix 0.5 -rb 

# Opening (bass line) notes of "Don't you want me" by The Human League (1981)
# shows how to specify integers and times of notes
# Integers are computed from closest i = f_s/(2*f), where f is the note
python audioorder.py -itp 200,200,268,225,200,200,268,225,169 -vtp 1,0.75,0.25,0.5,1.5,0.5,0.5,0.5,2.5 --beat 0.5 --noreverse -wf dywm.wav -nst 10 -stf traces_dywm.png -gs -sf spec_dywm.png -rb --mix 0.5
