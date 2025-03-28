# audibleintegers
Code for generating audio signals based on modular exponentiation of integers.

The integers show a surprising amount of structure, especially in the multiplicative group co-prime to a given integer N. This script can be used to generate audio tones that exhibit the repetitive patterns (via higher audio frequencies) for a given integer N.

Best results are for small-ish integers (< 1000.) Due to the 44.1 kHz sampling rate of audio, the approximate largest integer that can be fully "heard" is about 1000. No effort was made in time optimization so the script also gets slow as the integers get larger than 1000.

Clone this into a directory the usual way:

```
git clone https://github.com/jmoonware/audibleintegers
```

Best practice is to create a virtual environment:

```
python -m venv moonware
```

Then (on Linux):

```
source moonware/bin/activate
```

Windows:

```
moonware\bin\activate.bat
```

Then

```
cd audibleintegers
pip install .
````

That should do it.

Run

```
python audibleintegers.py -h
``` 

to see all the command line options.

Examples of output are in the subdirectory 'examples'. Run examples.sh to recreate similar files in the current directory.
