# to see all the command line options:
# python audioorder.py -h

# defaults
python audioorder.py 

# integers in order from 200-230, generating spectrogram and 
# plotting first 10 traces
python audioorder.py -sip 200 -nip 20 -gs -sf spec200.png -nst 10 -stf traces200.png -wf integers200.wav --mix 0.5

# integers in order from 200-230, generating spectrogram and 
# plotting first 10 traces, using random 'a' base
python audioorder.py -sip 200 -nip 20 -gs -sf spec200rb.png -nst 10 -stf traces200rb.png -wf integers200rb.wav --mix 0.5 -rb
