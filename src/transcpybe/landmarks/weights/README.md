# transcPybe v2: network weights and voicing maps

This folder is **not** under transcPybe's MIT licence. Its files are licensed under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International licence](https://creativecommons.org/licenses/by-nc-sa/4.0/)
(CC BY-NC-SA 4.0). The full text is in `LICENSE`, beside this file.

Copyright (c) 2026 David Whyatt.

In practice this means three things:

- **Attribution:** credit transcPybe, and the datasets below, when you use them.
- **NonCommercial:** no commercial use.
- **ShareAlike:** anything you build from them, such as fine-tuned weights, is shared under the same licence.

The code that runs the weights, in `transcpybe.landmarks`, is MIT like the rest of transcPybe.

## Why this licence

The networks learnt from singing that is licensed CC BY-NC-SA 4.0. The weights are an adaptation of it, so they carry
the same licence.

## What the files are

| File | What it is |
|---|---|
| `seed0.pt`, `seed1.pt`, `seed2.pt` | Three networks with the same design and training data, trained with seeds 0, 1 and 2. Their outputs are averaged. |
| `voicing-pesto.json`, `voicing-pyin.json` | Quantile maps from PESTO's and pYIN's confidence to CREPE full's, fitted on the training data's validation singers. |

Their sha256 hashes, which `tests/test_landmarks.py` checks:

```
b374fc0e7f0086a55a6a6f90fdec65e7c2d5d42df41f405d5152a1dd3d02852a  seed0.pt
47f2c5d14fb014a92608a8abac14a712e0370343098ece3000703f8f63068b92  seed1.pt
29f4237d7d66b2dcb1e675b9f52008536cd8c41adf0f2a97c1238ab4e5faf9a7  seed2.pt
cda7f2bd195f3e076f1a83987c7e001b9bf0b989faf582c5d9f11f6b41ebe7ba  voicing-pesto.json
55023bb82ca23fdc36c03a24b66c3e2874d850881efd90cefd353d3ed08eaaa1  voicing-pyin.json
```

### The networks

Each network is a `transcpybe.landmarks.model.BoundaryNet`:

- a convolutional front end over 80-band log-mel;
- a two-layer bidirectional GRU over time, with five pitch-track features alongside;
- 823,717 parameters;
- five outputs every 10 ms: where a syllable starts sounding, offset, sounding, re-sung onset, and where the
  syllable's vowel starts.

The networks were trained on Cambridge's CSD3 cluster (run `r2-lm`):

- 39,856 training clips and 5,241 validation clips;
- the validation singers held out of training;
- 30,000 steps.

Each file is the run's `best.pt`, copied unchanged. The best checkpoints came at steps 3,000 (seed 0), 4,000 (seed 1)
and 5,500 (seed 2).

The pitch features came from CREPE full.

### The voicing maps

Each JSON file is the training job's `voicing-map.json` (runs `r2-lm-pesto` and `r2-lm-pyin`), copied unchanged. It
holds the map and, on the validation frames it was fitted on, the share of frames voiced at 0.3 for CREPE full, for
the other tracker, and for the other tracker after the map.

## Training data, and its terms

The networks were trained on clips that put the vocals of two corpora over accompaniment from a third.

### Vocals: M4Singer

Licensed CC BY-NC-SA 4.0. Its licence asks users to cite:

> Zhang, L., Li, R., Wang, S., Deng, L., Liu, J., Ren, Y., He, J., Huang, R., Zhu, J., Chen, X., & Zhao, Z. (2022).
> M4Singer: A multi-style, multi-singer and musical score provided Mandarin singing corpus. *Advances in Neural
> Information Processing Systems 35* (Datasets and Benchmarks). https://doi.org/10.52202/068431-0501

### Vocals: GTSinger

Licensed CC BY-NC-SA 4.0, under the same terms. Its licence asks users to cite:

> Zhang, Y., Pan, C., Guo, W., Li, R., Zhu, Z., Wang, J., Xu, W., Lu, J., Hong, Z., Wang, C., Zhang, L., He, J.,
> Jiang, Z., Chen, Y., Yang, C., Zhou, J., Cheng, X., & Zhao, Z. (2024). GTSinger: A global multi-technique singing
> corpus with realistic music scores for all singing tasks. *Advances in Neural Information Processing Systems 37*
> (Datasets and Benchmarks). https://doi.org/10.52202/079017-0034

### Accompaniment: MUSDB18-HQ

Every stem but the vocals.

- MUSDB18-HQ is provided for educational purposes only, and is not to be used commercially without the copyright
  holders' permission.
- Some of its tracks carry their own licences. For example, its MedleyDB tracks are CC BY-NC-SA 4.0.

> Rafii, Z., Liutkus, A., Stöter, F.-R., Mimilakis, S. I., & Bittner, R. (2019). MUSDB18-HQ – an uncompressed
> version of MUSDB18. Zenodo. https://doi.org/10.5281/zenodo.3338373

### Changes we made

- The vocals were mixed with the accompaniment.
- Their note labels were converted into the networks' training targets.
- No audio from any of these corpora is distributed here.
