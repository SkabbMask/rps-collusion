# Figures

Detection, information analysis and completed model batches export figures
as 300-dpi PNG, SVG and PDF. Each figure directory includes numbered captions
and a manifest of source-file hashes.

## Rebuild saved results

```bash
python -m rps.figures --results results/full-detection-growth
python -m rps.figures --results results/full-information
python -m rps.figures --results results/qwen-hour
```

These commands read saved outputs. They do not collect games, contact models
or change predictions and calibration. Unchanged exports are reused; missing
files are rebuilt.

| Results directory | Figures |
| --- | --- |
| Detection output | `figures/paper_performance.*`, `paper_signal_and_horizon.*` |
| Information output | `figures/paper_information_bounds_01.png` and further groups |
| Model collection | `figures/paper_coordination_case_01.png` and further groups |

Captions are `detection_captions.md`, `information_captions.md` or
`collection_captions.md` in the corresponding `figures/` directory. Model
analyses also have figures in their `detection/` and `information/` subfolders.

## Reading the figures

- Titles distinguish scripted random players from recorded model games.
- Performance plots use pointwise Wilson 95% intervals. Recovery counts misses
  as failures. Different thresholds do not constitute a matched-FPR comparison.
- Tiny model test sets are illustrations. Infinite cutoffs explain zero flags
  in the six-game Qwen check; they do not establish failure of the raw score.
- Information plots show lower bounds through a fixed feature map and detector.
  Zero does not establish zero information, and bounds do not rank true views.
- Model-case plots select the first held-out coalition game per group in
  manifest order. With several helpers they show the first helper.

The exact toy model was inconclusive for the paper's empirical question and
was excluded. Default exports omit it, including when rebuilding old results.
Its optional diagnostic can be requested with `--include-exact`; it is not a
paper result. Old generated diagnostic files remain as historical artifacts.
