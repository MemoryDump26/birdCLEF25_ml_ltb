
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

8kHz failed really hard, the upper split basically doesn't work at all (very low AUC in training). Maybe a split at like 4kHz would be better?

## EfficientNet B0 input shape

A lot of documentations said that the input shape is 224x224, but the target shape in the baseline notebook is 256x256? Well probably doesn't matter, the score still went down XD

The resampling method doesn't change anything, lanczos just make the background a bit brighter.
## Higher n_mels

When n_mels > 256 and n_fft = 1024, gaps in frequency start to show. 
# Score

| Models                                                                                         | LB score | Notes                                                                           |
| ---------------------------------------------------------------------------------------------- | -------- | ------------------------------------------------------------------------------- |
| Baseline (5 folds)                                                                             | 0.761    |                                                                                 |
| Baseline, no voice (5 folds)                                                                   | 0.793    | Removing human voices helped a lot                                              |
| Baseline, no voice, PCEN, 128 hop_length (5 folds)                                             | 0.786    | PCEN doesn't really works (4th place last year tried)                           |
| Baseline, no voice, 128 hop_length, mu_expand noise filtering (5 folds)                        | 0.811    | This is huge XD                                                                 |
| Baseline, no voice, 128 hop_length, mu_expand noise filtering (1 folds)                        | 0.783    | **New baseline**                                                                |
| 8kHz split (3 low, 3 high ensemble)                                                            | 0.741    | XDDDD, the high split is basically useless, maybe lowering the split will help. |
| Baseline, 224x224 no scaling (1 fold)                                                          | 0.777    |                                                                                 |
| Baseline, no augmentation (1 fold)                                                             | 0.785    |                                                                                 |
| Baseline, brightness augmentation only (1 fold)                                                | 0.777    | ????? what ?????                                                                |
| Baseline, XY masking augmentation only (1 fold)                                                | 0.775    | this doesn't make any sense XDDD                                                |
| Baseline, no augmentation, mixup alpha = 1 (1 fold)                                            | 0.785    |                                                                                 |
| Baseline, no augment, batch_size=16, mixup_alpha=0.2 (1 fold)                                  | 0.784    |                                                                                 |
| Baseline, no augment, batch_size = 8, mixup_alpha = 0.2 (1 fold)                               | 0.782    |                                                                                 |
| Same as above, fft = 2048 (1 fold)                                                             | 0.755    | XDD Val AUC is 0.87 right from epoch 1, pretty sus                              |
| fft = 2048, batch_size=32, default augment, mixup_alpha = 0.4 (basically all default) (1 fold) | 0.809    | ??????????? why ???????? oh well<br>**New baseline**                            |
|                                                                                                |          |                                                                                 |

# Things to try
- Tweaking augmentation probabilities (XY masking, brightness)
- Submission CSV post-processing (prediction smoothing by averaging adjacent rows)
	- Example: `def smooth_submission()` in this [notebook](https://www.kaggle.com/code/tomkkk/change-secondary-labels-in-train-csv)
- Optimizer & scheduler hyper-parameters.
- Test time augmentation
- Overlapping inference window
