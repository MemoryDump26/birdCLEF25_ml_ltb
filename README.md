# Problems

## Audio recordings contain human voices

The following [baseline model](https://www.kaggle.com/code/kadircandrisolu/efficientnet-b0-pytorch-train-birdclef-25/notebook) [generates mel-spectrogram](https://www.kaggle.com/code/kadircandrisolu/transforming-audio-to-mel-spec-birdclef-25) by taking 5 seconds of audio right at the middle of recordings in `train_audio`. However, many recordings only contain around 10 seconds of actual animal audio (often at the beginning), and the rest (often a minute long) is human voices. This means that their [precomputed spectrogram](https://www.kaggle.com/datasets/kadircandrisolu/birdclef25-mel-spectrograms/data) have some classes that only contain human voices spectrogram (e.g. 1139490).

In this [notebook](https://www.kaggle.com/code/kdmitrie/bc25-separation-voice-from-data/notebook), every section of human speeches is timestamped. We can use this information to avoid human speeches when generating spectrograms.

[Someone did the work already XDD](https://www.kaggle.com/code/verniy73/transforming-audio-to-mel-spec-without-human-voice)

# Experiments

## Mel-spectrogram config

Higher FFT window size lead to more resolution in the frequency domain, but at the cost of significantly worse time-domain resolution (separate sounds blur together). This can be quite detrimental for rhythmic features like birdsongs.

With 2048, the spectrogram is somewhat blurred, and from 4096 onwards, the transients all overlapped each other, unrecognizable.

The `hop_length` parameter essentially adjust the FFT window overlap, smaller `hop_length` result in higher overlap, which is better for time resolution. Visual comparison between 128 and 512 `hop_length`, at 1024 FFT window size, we can see the spectrogram with 128 `hop_length` contain much sharper transients.

For now 1024 FFT window size with 128 hop_length seems to be the best (64 hop_length doesn't really change anything). 512 FFT window size only adds gaps in the lower frequencies.

Using `power=1.0` and `amplitude_to_db` seems to reduce the background brightness on some signal only, weird.

## librosa nn_filter

Denoise by nearest-neighbor. Way too blurry lol.

## PCEN

Looks promising at first, but scored lower lol. The foreground elements are not bright enough I think.
## librosa.mu_expand

Basically remove weak signal from the audio. The resulting spectrogram looked really nice, background noise is basically gone, but foreground audio is kept the same, maybe even brighter.

## Split the spectrogram into bands, then train and inference on each


# Score

| Models                                                        | LB score | Notes                                                 |
| ------------------------------------------------------------- | -------- | ----------------------------------------------------- |
| Baseline                                                      | 0.761    |                                                       |
| Baseline, no voice                                            | 0.793    | Removing human voices helped a lot                    |
| Baseline, no voice, PCEN, 128 hop_length                      | 0.786    | PCEN doesn't really works (4th place last year tried) |
| Baseline, no voice, 128 hop_length, mu_expand noise filtering | 0.811    | This is huge XD                                       |
