# Next project: noise-to-clean training on synthetic data

Hand-off brief for a new chat. Goal: build a **synthetic training set** of
(noisy, clean) image pairs whose noise is **sampled from the noise of our own CROCUS
radiographs**. Then train a supervised denoiser (noisy → clean) and check whether it keeps
fine detail better than the current Noise2Noise model.

## 1. Where we stand (from the n2nni repo)

Full report: `n2nni/report/build/main.pdf`.

- **Best model so far (R6):** 2D dilated CNN (333 k params, receptive field 97 px).
  - Trained with Noise2Noise (N2N), annealed L0 loss, on raw ADU sums of frames, with one global intensity scale.
  - At inference: 8-fold rotation/flip averaging.
  - Checkpoint: `n2nni/runs/dc399383-6480-49ad-8270-1893d36b20a1/train/best.pt`
  - Inference code: `n2n_train/denoise.py` on branch `orx/r6-...`
- **Noise:** 5 input frames match about **50–65 stacked frames**, roughly 10× less beam time. No transmission bias.
- **Problem: blur.** On the PSI test pattern (60 s):
  - edge FWHM is 10.75 px, against 7.3 px for raw 5 frames and 5.3 px for the reference;
  - medium/coarse strip contrast keeps only 16% / 30% of the reference;
  - the finest strips are lost.
- **Why N2N blurs:** at low SNR the fine strips can't be told apart from splash noise locally, so N2N regresses to the mean. The training data also has almost no fine, sharp structure; the PSI pattern is held out.

## 2. Lessons that must shape the synthetic data

- **The noise is not Gaussian and not pixel-independent.**
  - Each neutron gives a light splash over several pixels, so the noise is spatially correlated.
  - Splash brightness varies.
  - Sparse gamma white spots hit single frames.
  - Blind-spot methods (Noise2Void, N2V2) therefore fail.
- **A Gaussian correlated noise model already failed.** Noisier2Noise used variance = g·signal + read noise (g ≈ 1.13) with the measured correlation kernel. Result: ncmse 3.4e-2 vs 9e-4 for R6, and tens of thousands of outliers. **Realistic noise is the whole game.**
- **The noise is signal-dependent.** Dark strips (low transmission) must get the right, lower count noise.
- **No filtering of the raw data.** The user wants the ML to handle gamma spots, not a pre-filter.
- **Keep transmission quantitative.** Any bias in mean T loses, even with good PSNR.

## 3. Plan

### Step 1: clean structures (ground truth)
- Library of transmission maps T(x, y):
  - strips at many pitches (2–40 px) and angles
  - edges and steps
  - dots/discs
  - smooth gradients, with mixed absorption levels
- Blur by the **detector PSF measured on the Cd edge**, so the target is what a noise-free detector would record. No deconvolution, to avoid hallucination.

### Step 2: realistic noise, sampled from our own images
- **Option A (preferred): event-level simulation.**
  - Neutron hits are Poisson with rate ∝ T·flux.
  - Each hit pastes a **real splash** cut out of low-flux 10 s frames, keeping its shape and brightness spread.
  - Add read noise and the DC offset, taken from real DC frames.
  - Add **gamma spots sampled from real frames**.
- **Option B (fallback): bank of real noise.**
  - Noise = frame − cleaned stack mean, from long OB stacks (60 frames).
  - Match it to the local signal level using OB stacks at several exposures (10/60/150/300 s).

### Step 3: validate the simulator before any training
A simulated OB stack must match a real one on:
- variance vs mean (gain ≈ 1.13 ADU²/ADU, plus the intercept)
- the pixel-to-pixel correlation / noise power spectrum
- histogram tails (splashes, gamma spots)
- frame differences that look the same by eye

**Stop if it doesn't match.** Show a real-vs-simulated comparison figure first.

### Step 4: train noisy → clean
- Plain L2 against the clean target; same architecture as R6 for a fair comparison.
- Optionally mix in real N2N pairs to stay anchored to real data.
- Input: raw ADU sums of k frames (k = 1..5), with the same scaling as R6.

### Step 5: evaluate with the same protocol as n2nni
- The input is 5 frames; the reference is the cleaned mean of the remaining frames.
- **VAL** (selection only): NI_12_08 `cd_tilted`, `alu_fine_steps`.
- **Test:** the whole NI_03_07 session, and PSI 60 s. PSI 10 s may be used as a detail-validation scene.
- **Metrics:** ncmse, bias, edge FWHM, PSI line-pair contrast (g1–g3), 5σ outliers.
- **Watch for hallucination:** check that the model does not invent strips in flat areas (residual images, outliers).
- **Baseline to beat:** R6 (VAL ncmse 1.44e-4; PSI 60 s edge 10.75 px; g2 0.013, g3 0.029).

## 4. Data and environment

- **Raw data:** `/home/lrs/projects/n2nni/data_ni_crocus`
  - 2048×2048 FITS, int16 with BZERO, ~100 µm pixels.
  - **Never copy, modify or commit raw data.**
- **Catalog:** `n2nni/catalog/manifest.json`, regenerate with `python -m n2n_data.build`.
  - Stacks are grouped by object / OB / DC, with fixed train/val/test splits.
  - Reader: `n2n_data/store.py` (`FrameStore`).
- **Useful long stacks:** NI_03_07 OB (60 frames), PE steps and water samples (60 frames each), the NI_12_08 val stacks (20 frames each). The PSI 10 s stack (20 frames) is useful for extracting splashes.
- **Machine:** RTX 5070, 12 GB.
- **Conda env:** `n2nni-env`, Python 3.13, torch 2.14 with cu130.

## 5. Working preferences

- Ask before installing or downloading anything.
- No git commands. The user commits and pushes themselves; just tell them the steps.
- Never commit images, arrays, weights, logs or JSON. Only code, config, `.md` and the PDF report.
- The user runs long trainings in their own tmux. Give the command rather than launching long jobs, unless asked.
- Show normalized transmission images, `(obj − DC)/(OB − DC)`, with `origin="lower"`, and give each image its own gray window when judging detail.
- Ask when something is unclear, and keep explanations short.
