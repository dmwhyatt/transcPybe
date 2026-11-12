"""A learned note-boundary detector for transcPybe, and the pipeline that trains it.

Not part of the installed package; what it trained ships as ``transcpybe.landmarks``.
The stages, each a module with its own command line, run in this order:

    sources   labelled singing -> a manifest of vocals with their notes
    mix       each vocal over an accompaniment, as a pop mix would have it
    separate  each mix through the lead-vocal separator the pipeline uses
    features  each separated stem -> log-mel, pitch features and frame targets
    train     a small network that marks where notes start and end
    infer     its onset posterior as a transcPybe OnsetTrack

See boundary/README.md.
"""

HOP_S = 0.01            # one frame per transcPybe f0 frame: 160 samples at 16 kHz
FEATURE_SR = 16000      # what transcPybe analyses at
MIX_SR = 44100          # what the separator is trained on
