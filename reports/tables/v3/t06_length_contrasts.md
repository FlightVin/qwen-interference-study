# Length extension: paired contrasts at both caps

Paired contrasts of pass@1 before (12,288) and after (16,384) the extension, and the change of the contrast. R − H and U − H exist for seed 101 only (H-s202 was not extended). Interval: paired problem bootstrap (audit pass B).

Sources: `research/audit-a26/pass-b/data/eval_length.json`.

## Sealed test

| Contrast | At 12,288 (points) | 95% interval | At 16,384 (points) | 95% interval | Change (points) | 95% interval |
|---|---|---|---|---|---|---|
| R-s101 − U-s101 | +4.2 | [+2.0, +6.3] | +3.6 | [+1.5, +5.7] | −0.6 | [−1.6, +0.4] |
| R-s202 − U-s202 | −4.4 | [−7.3, −1.8] | −3.9 | [−6.7, −1.3] | +0.5 | [−0.3, +1.3] |
| R-s101 − H-s101 | +4.9 | [+2.1, +7.9] | +4.9 | [+2.1, +7.9] | 0.0 | [−0.8, +0.9] |
| U-s101 − H-s101 | +0.7 | [−1.9, +3.6] | +1.3 | [−1.0, +4.0] | +0.6 | [−0.2, +1.5] |

## Stress panels

| Contrast | At 12,288 (points) | 95% interval | At 16,384 (points) | 95% interval | Change (points) | 95% interval |
|---|---|---|---|---|---|---|
| R-s101 − U-s101 | +4.2 | [+1.5, +6.8] | +4.1 | [+1.4, +6.7] | −0.1 | [−1.1, +1.0] |
| R-s202 − U-s202 | −2.7 | [−5.0, −0.2] | −2.1 | [−4.6, +0.3] | +0.5 | [−0.5, +1.5] |
| R-s101 − H-s101 | +5.4 | [+2.6, +8.1] | +5.0 | [+2.3, +7.7] | −0.3 | [−1.5, +0.8] |
| U-s101 − H-s101 | +1.2 | [−1.5, +3.8] | +1.0 | [−1.9, +3.7] | −0.2 | [−1.5, +0.9] |

Rounding: every displayed number is rounded to one decimal independently from the unrounded value in the JSON file. A displayed difference can therefore differ by 0.1 from the difference of the displayed scores; the unrounded contrast is the correct one.
