
# Problems

## Audio recordings contain human voices

The following [baseline model](https://www.kaggle.com/code/kadircandrisolu/efficientnet-b0-pytorch-train-birdclef-25/notebook) [generates mel-spectrogram](https://www.kaggle.com/code/kadircandrisolu/transforming-audio-to-mel-spec-birdclef-25) by taking 5 seconds of audio right at the middle of recordings in `train_audio`. However, many recordings only contain around 10 seconds of actual animal audio (often at the beginning), and the rest (often a minute long) is human voices. This means that their [precomputed spectrogram](https://www.kaggle.com/datasets/kadircandrisolu/birdclef25-mel-spectrograms/data) have some classes that only contain human voices spectrogram (e.g. 1139490).

In this [notebook](https://www.kaggle.com/code/kdmitrie/bc25-separation-voice-from-data/notebook), every section of human speeches is timestamped. We can use this information to avoid human speeches when generating spectrograms.

[Someone did the work already XDD](https://www.kaggle.com/code/verniy73/transforming-audio-to-mel-spec-without-human-voice)
## Best AUC doesn't correlate to best model


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

The resampling method doesn't seems to change anything, lanczos just make the background a bit brighter.
## Higher n_mels

When n_mels > 256 and n_fft = 1024, gaps in frequency start to show. 
## Prediction temporal smoothing
By averaging adjacent rows (coresponding to 5 seconds soundscape chunks that are adjacent to each other), the prediction is more stable over time. Increase LB score by ~0.015, huge XD
## Pseudo labelling
Included in the dataset is `train_soundscapes`, containing unlabeled 1 minute soundscapes, similar to those in the validation set. By cutting these up and label them, we can gain more data to train newer models on.
# Score

| Models                                                                                          | AUC                     | LB score | Notes                                                                                                         |
| ----------------------------------------------------------------------------------------------- | ----------------------- | -------- | ------------------------------------------------------------------------------------------------------------- |
| Baseline (5 folds)                                                                              |                         | 0.761    |                                                                                                               |
| Baseline, no voice (5 folds)                                                                    | 0.9473                  | 0.793    | Removing human voices helped a lot                                                                            |
| Baseline, no voice, PCEN, 128 hop_length (5 folds)                                              | 0.9460                  | 0.786    | PCEN doesn't really works (4th place last year tried)                                                         |
| Baseline, no voice, 128 hop_length, mu_expand noise filtering (5 folds)                         | 0.9448                  | 0.811    | This is huge XD                                                                                               |
| Baseline, no voice, 128 hop_length, mu_expand noise filtering (1 folds)                         | 0.9427                  | 0.783    | **New baseline**                                                                                              |
| 8kHz split (3 low, 3 high ensemble)                                                             | ---                     | 0.741    | XDDDD, the high split is basically useless, maybe lowering the split will help.                               |
| Baseline, 224x224 no scaling (1 fold)                                                           | 0.9414                  | 0.777    |                                                                                                               |
| Baseline, no augmentation (1 fold)                                                              | 0.9375                  | 0.785    |                                                                                                               |
| Baseline, brightness augmentation only (1 fold)                                                 | 0.9415                  | 0.777    | ????? what ?????                                                                                              |
| Baseline, XY masking augmentation only (1 fold)                                                 | 0.9452                  | 0.775    | this doesn't make any sense XDDD                                                                              |
| Baseline, no augmentation, mixup alpha = 1 (1 fold)                                             | 0.9375                  | 0.785    |                                                                                                               |
| Baseline, no augment, batch_size=16, mixup_alpha=0.2 (1 fold)                                   | 0.9452                  | 0.784    |                                                                                                               |
| Baseline, no augment, batch_size = 8, mixup_alpha = 0.2 (1 fold)                                | 0.9332                  | 0.782    |                                                                                                               |
| Same as above, fft = 2048 (1 fold)                                                              | 0.9416                  | 0.755    | XDD Val AUC is 0.87 right from epoch 1, pretty sus                                                            |
| fft = 2048, batch_size=32, default augment, mixup_alpha = 0.4 (basically all default) (1 fold)  | 0.9422                  | 0.809    | ??????????? why ???????? oh well<br>**New baseline**                                                          |
| 4096_fft, 256_mel, hop mismatch (16 train, 128 inference)                                       | 0.9434                  | 0.785    |                                                                                                               |
| 4096_fft, 512_mel&hop, 2 pass mu_expand = 63                                                    | 0.9441                  | 0.788    | Best AUC at epoch 9, might be overfitting?                                                                    |
| Same as above, batch_size = 16                                                                  | 0.9435                  | 0.756    | XDDD nah                                                                                                      |
| Same as above, batch_size = 64                                                                  | 0.9397                  | 0.785    | welp batch_size = 32 still da best                                                                            |
| Same as above, with prediction temporal smoothing                                               | ---                     | 0.798    | !!!! **wowowowow** !!!!                                                                                       |
| Baseline, 4096_fft, 256_mel&hop, 2 pass mu_expand = 63, batch_size = 32 (no temporal smoothing) | 0.9445                  | 0.788    | welp 2048 fft still da best                                                                                   |
| 8096fft, 512hop, 128mel, 2x127muExpand                                                          | 0.9396                  | 0.749    | forgot to update to the newest model !!!!! But even after updating, the score stays the same @@ why           |
| 2048fft, 626hop (noScale), 256mel, 2x31muExpand (stronger denoise)                              | 0.9458                  | 0.789    | Maybe some noise is good? 256mel might be bad too                                                             |
| Same as above, with mel_spec values clipping (50, 99.5)                                         | 0.9389                  | 0.791    | hmmm                                                                                                          |
| Same as above, with SWA (checkpoint averaging) 10 epochs, swa: 6start 0.0005lr 2anneal          | 0.9389 best, 0.9421 SWA | 0.784    | feels bad man                                                                                                 |
| 20 epochs, SWA: 10start 0.00025lr 5anneal                                                       | 0.9399 best, 0.9373 SWA | 0.758    | XD                                                                                                            |
| Back to baseline mel_spec, with SWA (same as above)                                             | 0.9421 best, 0.9439 SWA | 0.733    | ????? forgot to update mel_spec params in inference !!!!!                                                     |
| Correct mel_spec param                                                                          | ---                     | 0.772    | meh XDD                                                                                                       |
| swa: 8start, 0.00025lr, 1anneal                                                                 | 0.9421 best, 0.9420 SWA | 0.765    | XDDD<br>The example code from Pytorch count epoch from 0, 8start means "from epoch 9", the last one @@.       |
| swa: 6start, 2anneal. Bump swa lr to 0.0025 (one less zeros)                                    | 0.9421 best, 0.9453 SWA | 0.783    | huh, higher is better? Nah it's probably because there's actually 3 models being averaged together this time. |
| bump swa lr to 0.025 (one less zeros)                                                           | 0.5935 SWA              | ---      | model seems to diverge at lr=0.0126. 0.001 seems to be a good values                                          |
| swa lr = 0.001                                                                                  | 0.9457 SWA              | 0.773    |                                                                                                               |
| swa start epoch fix; start5, 0.001lr, 3anneal                                                   | 0.9406 best, 0.9483 AUC | 0.786    |                                                                                                               |
| swa 0.00025lr                                                                                   | 0.9409 best, 0.9456 SWA | 0.778    |                                                                                                               |
| back to baseline, with clipping and FocalLossBCE                                                | 0.9407                  | 0.777    | vv huh so focal loss actually improve score? and swa too???                                                   |
| baseline with clipping, sanity check                                                            | 0.9421                  | 0.765    | man it's probably the clipping XDDD                                                                           |
| real baseline this time frfr                                                                    | 0.9447                  | 0.802    | okay finally XD                                                                                               |
| real baseline with swa: 5start, 0.001lr, 3anneal                                                | 0.9399 best, 0.9493 SWA | 0.780    | sighhhhhhh                                                                                                    |
| real baseline with swa and focal loss                                                           | 0.9454 best, 0.9512 SWA | 0.786    |                                                                                                               |
| baseline with focal loss only                                                                   | 0.9448                  | 0.776    | sighhhhh XDDD                                                                                                 |
| baseline 5 fold, with smoothing                                                                 | 0.9453 mean             | 0.817    |                                                                                                               |
| baseline 5 fold, smoothing and fixed lerp, even matching epoch                                  | ---                     | ---      | same thing, LB score stuck at 0.8                                                                             |
| adam 0.01 weight decay                                                                          | 0.9404                  | 0.799    |                                                                                                               |
| RAdam optimizer (implicit LR warmup)                                                            | 0.9414                  | 0.799    |                                                                                                               |
| RAdam with original weight decay                                                                | 0.9444                  | 0.801    | XD AUC is so useless man XD                                                                                   |
| RAdam, epoch 15 (last)                                                                          | 0.9429                  | 0.781    |                                                                                                               |
| RAdam, epoch 11 (best)                                                                          | 0.9441                  | 0.787    |                                                                                                               |
| RAdam, epoch 8 (early)                                                                          | 0.9425                  | 0.809    | overfitting XDD the scheduler make tuning epochs kinda annoying                                               |
| 8 epochs, FocalLoss, SWA: 5start, 0.0003lr, 3anneal                                             | 0.9425 best, 0.9449 SWA | 0.804    |                                                                                                               |
| Same as above, without FocalLoss                                                                | 0.9420 best, 0.9441 SWA | 0.810    |                                                                                                               |
| SWA: 6start, 5 fold                                                                             |                         | 0.818    |                                                                                                               |
| Same as above, fold 1 swa only                                                                  |                         | 0.811    |                                                                                                               |
| Same as above, 5 fold (no swa)                                                                  |                         | 0.817    |                                                                                                               |
| No kfold, swa                                                                                   | ---                     | 0.798    | last epoch train AUC: 0.9793                                                                                  |
| No kfold, no swa, 15 epoch (submit epoch 10)                                                    | ---                     | 0.776    | train AUC: 0.9918                                                                                             |
| No kfold, no swa, 15 epoch (submit epoch 7)                                                     | ---                     | 0.766    | train AUC: 0.9677                                                                                             |
| No kfold, no swa, 15 epoch (submit epoch 15)                                                    | ---                     | 0.782    | train AUC: 0.9983                                                                                             |
| Same as above, CrossEntropyLoss                                                                 | ---                     | 0.737    |                                                                                                               |
| 8 epochs, FocalLoss with equal weights, SWA: 5 start, 0.0003lr, 3anneal                         | 0.9400 best, 0.9437 SWA | 0.811    | hmmm, not bad                                                                                                 |
| different temporal smoothing method (with global average), TTA                                  | ---                     | 0.772    |                                                                                                               |
| new augments, 12 epochs swa 7, 0.001, 3                                                         | 0.9476 best, 0.9480 SWA | 0.796    | Val AUC crashed hard at epoch 5 & 6?                                                                          |
| reduced aug probs, weaker clipping                                                              | 0.9391 best, 0.9413 SWA | 0.786    |                                                                                                               |
| 0.5 all aug probs, no clipping                                                                  | 0.9471 best, 0.9473 SWA | 0.816    | wow, clipping is really bad XDDD                                                                              |
| 0.4 all aug, no clipping, old smoothing (no average)                                            |                         | 0.813    |                                                                                                               |
| noKFold, rotated seed & augment (5 ensemble)                                                    | ---                     | 0.817    | Train Loss: 0.0105, Train AUC: 0.9899                                                                         |
| 5 fold, different augment prob                                                                  | ---                     | 0.812    |                                                                                                               |
| 5 fold, 0.5 all aug prob                                                                        | 0.9493 mean             | 0.818    |                                                                                                               |
| efficient net v2s backbone, 3/5 fold                                                            | 0.9546 mean             | 0.818    |                                                                                                               |
| same as above, rms segment, 10 epoch                                                            | 0.9616 mean             | 0.815    | high AUC low LB score XDDD                                                                                    |
| same as above, but 4/5 fold                                                                     | ---                     | 0.819    |                                                                                                               |
| efficient net b0 backbone, rms segment, 5 fold                                                  | 0.9558 mean             | 0.815    |                                                                                                               |
| same as above, disable mu_expand noise filtering in inference                                   | ---                     | 0.819    |                                                                                                               |
| same as above, enable mu_expand, disable temporal smoothing                                     | ---                     | 0.802    |                                                                                                               |

# Things to try
- Tweaking augmentation probabilities (XY masking, brightness)
- Submission CSV post-processing (prediction smoothing by averaging adjacent rows)
	- Example: `def smooth_submission()` in this [notebook](https://www.kaggle.com/code/tomkkk/change-secondary-labels-in-train-csv)
- Optimizer & scheduler hyper-parameters.
- Test time augmentation
- Overlapping inference window

- 3 input channels: `timm` already modified the model to accomodate for 1 channel input, but [3 channel input](https://towardsdatascience.com/transfer-learning-on-greyscale-images-how-to-fine-tune-pretrained-models-on-black-and-white-9a5150755c7a/) might yield better result.
- [Focal loss BCE](https://www.kaggle.com/code/hideyukizushi/bird25-onlyinf-v2-s-focallossbce-cv-962-lb-829)
