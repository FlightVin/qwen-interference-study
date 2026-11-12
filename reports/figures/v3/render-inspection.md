# Render inspection of `reports/REPORT.md` v3 (A26 §6)

This file records two render checks. Render 2 checks the redesigned figures and replaces the verdicts of Render 1.
Render 1 is kept below it as the record of the first check.

## Render 2: redesigned figures (clone `1e75db1`)

Purpose: repeat the Render 1 check on the redesigned figures. Result: all 14 figures pass at both widths. The smallest
figure text is 21.9 px at normal width and 10.4 px at 420 px, so every label is legible without zoom on a phone.
Caption numbers run 1–14 in page order. All 4 "Figure N" cross-references point at the right figure. Plotted
values agree with the tables in all 47 spot checks. The seven Render 1 fixes are all applied. One optional item
remains: most figures are now taller than one laptop screen at normal width (section R2.6).

Date: 2026-11-11 (EST). Inspector: agent "render".

### R2.1 What was rendered

| Item | Value |
|---|---|
| Source | new clean clone of `git@github.com:FlightVin/qwen-interference-study.git`, branch `main` |
| Clone SHA | `1e75db12cbe4edd06055c9d11954e8197df873d8` ("Report v3 figures: narrow-screen redesign …") |
| Renderer | GitHub API `POST /markdown`, `mode=markdown`, `context=FlightVin/qwen-interference-study` |
| Page | the CSS of Render 1 (system font 16 px, column `max-width:830px; padding:24px`, `img{max-width:100%}`, tables `display:block; overflow:auto`) |
| Method | one page per figure: the paragraph before the figure, the figure, and its caption, cut from the rendered HTML |
| Normal width | headless Chrome window 1024 px; content column 782 px |
| Narrow width | 420 px viewport (iframe of width 420 in a 520 px window, shot cropped to the iframe); content column 372 px |

The method is the same as Render 1 section R1.1. Scrolling to a `#figN` fragment gave blank shots in headless
Chrome, and Chrome on macOS has a 500 px minimum window width.

### R2.2 Numbering and cross-references

Each caption number matches the figure's position in the page.

| Page position | File | Caption | Section |
|---|---|---|---|
| 1 | `f01_endpoints.png` | Figure 1 | 5.1 |
| 2 | `f02_contrasts_test.png` | Figure 2 | 5.1 |
| 3 | `f03_dev_rounds.png` | Figure 3 | 5.2 |
| 4 | `f04_dev_hard_attempts.png` | Figure 4 | 5.2 |
| 5 | `f05_traj_JH_JR.png` | Figure 5 | 5.2 |
| 6 | `f06_traj_JH_JU.png` | Figure 6 | 5.2 |
| 7 | `f12_control.png` | Figure 7 | 5.3 |
| 8 | `f07a_delta_j_r0.png` | Figure 8 | 5.4.4 |
| 9 | `f07b_delta_j_r64.png` | Figure 9 | 5.4.4 |
| 10 | `f08_length_pass1.png` | Figure 10 | 5.5 |
| 11 | `f08b_length_capped.png` | Figure 11 | 5.5 |
| 12 | `f09_length_contrasts.png` | Figure 12 | 5.5 |
| 13 | `f10_stress_tiers.png` | Figure 13 | 5.6 |
| 14 | `f11_stress_contrasts.png` | Figure 14 | 5.6 |

`figures/v3/captions.md` uses the same numbering. The file names keep the old numbers, for example `f12_control.png`
is Figure 7. Readers do not see file names, so this does not affect the report.

A script listed every "Figure N" in `REPORT.md`. Apart from the 14 captions, there are 4 cross-references:

| Line | Text | Points at | Correct |
|---|---|---|---|
| 264 | Figure 1 caption: "compare policies with the paired contrasts in Figure 2" | Figure 2, sealed-test paired contrasts | yes |
| 335 | Figure 4 caption: "Markers as in Figure 3; no intervals drawn" | Figure 3, same markers; draws no intervals | yes |
| 381 | Figure 6 caption: "As Figure 5, with y: J_U" | Figure 5, the (J_H, J_R) paths | yes |
| 522 | Figure 9 caption: "As Figure 8, from R-s101 round 64" | Figure 8, the same diagnostic from round 0 | yes |

The body text has no other "Figure", "Fig." or lowercase "figure N" references.

### R2.3 Rendered text size

All figures are 6.0 in wide at 200 dpi (1188–1189 px). Tick labels and legends use 12 pt, axis labels 13 pt, and
panel titles 14 pt (`build_v3.py` lines 561–574). The only other text sizes in the script are annotations at 12 and
13 pt.

| Text | 1024 px window (scale 0.658) | 420 px viewport (scale 0.313) |
|---|---|---|
| Tick labels, legend, arm letters (12 pt) | 21.9 px | 10.4 px |
| Axis labels (13 pt) | 23.8 px | 11.3 px |
| Panel and figure titles (14 pt) | 25.6 px | 12.2 px |

The table gives font sizes in CSS px at device scale 1. The narrow screenshots confirm that every tick and legend
entry is legible at 100%.

### R2.4 Verdicts per figure and width

Pass means: title and axis labels not clipped, text legible at 100% zoom, no overlapping text, legend clear of data,
caption directly below and consistent with the figure.

| Fig. | File | 1024 px | 420 px | Notes |
|---|---|---|---|---|
| 1 | f01 | pass | pass | Single panel. |
| 2 | f02 | pass | pass | At 420 the x ticks "−7.5 −5.0 −2.5" sit close together but do not touch. |
| 3 | f03 | pass | pass | Panels stacked (seed 101 over seed 202) with a shared x axis. Legend below, clear of data. |
| 4 | f04 | pass | pass | Caption now says "no intervals drawn". |
| 5 | f05 | pass | pass | Round-128 markers are larger, black-edged and labelled H/R/U. "r0" sits left of the start points. The seed-202 H and R ends are still close together, but their letters separate them. |
| 6 | f06 | pass | pass | As f05. In seed 101 the R round-96 and round-128 markers overlap, because their J_U values are equal (96.1). This reflects the data. |
| 7 | f12 | pass | pass | Three stacked panels with one x axis. The "J" of the "J_U" and "J_R" panel titles touches the y-axis line (cosmetic). |
| 8 | f07a | pass | pass | Round-0 diagnostic alone. Legend has no momentum-only entry; none occurs at round 0. |
| 9 | f07b | pass | pass | Momentum-only draws are crossed markers, explained in legend and caption. |
| 10 | f08 | pass | pass | R-s202 open markers now clear the y axis. |
| 11 | f08b | pass | pass | n is in each row label. The x label reads "Share of the capped attempts (%)". |
| 12 | f09 | pass | pass | |
| 13 | f10 | pass | pass | Panels stacked; the x label is centred under the panels. |
| 14 | f11 | pass | pass | SuperHard U − H seed 202 upper end (+11.1) now fits inside the axis. |

The three tables at 420 px (§5.1 endpoint table, §5.5 length table, §5.6 stress table) pass, as in Render 1. Each
scrolls sideways inside the content column; the page itself does not scroll sideways.

**Render 1 fixes.** All seven are applied. 1: page-order numbering. 2: f10 x label centred. 3: f08 markers clear the
axis line. 4: f05/f06 ends labelled and "r0" moved. 5: f07 split into f07a and f07b. 6: Figure 4 caption corrected.
7: f08b x label reworded. Item 8 (phone legibility) is solved by the redesign, not by a tap-to-zoom note.

### R2.5 Value spot checks (plot against table)

Values were read off the 1024 px shots, to about ±0.5 point (±1 point on the 0–100 bar scale of Figure 11).

| Fig. | Quantity | Read off plot | Table value (source) | Agree |
|---|---|---|---|---|
| 1 | base pass@1 | ≈52.6 | 52.6 (`t01`) | yes |
| 1 | R-s101 r128 | ≈82 | 81.9 (`t01`) | yes |
| 1 | H-s202 r128 | ≈72.4 | 72.4 (`t01`) | yes |
| 2 | R − U seed 101 | ≈+4.2 | +4.2 (`t02`) | yes |
| 2 | R − U seed 202 | ≈−4.4 | −4.4 (`t02`) | yes |
| 2 | U − H seed 202, interval | ≈+6.8, ≈3.6 to 10.2 | +6.8 [+3.6, +10.2] (`t02`) | yes |
| 3 | R-s101 r64 dev | ≈73.6 | 73.6 (`t03`) | yes |
| 3 | U-s101 r32 dev | ≈60 | 60.1 (`t03`) | yes |
| 3 | U-s202 r128 dev | ≈70.3 | 70.3 (`t03`) | yes |
| 4 | H-s101 at 16,384 attempts | ≈66.3 | 66.3 (`t03`) | yes |
| 4 | R-s101 at 6,144 (r96) | ≈74.8 | 74.8 (`t03`) | yes |
| 4 | H-s202 at 4,096 (r32) | ≈56.6 | 56.6 (`t03`) | yes |
| 5 | H-s101 r128 (J_H, J_R) | ≈(69, 89.3) | (68.8, 89.3) (`t03`) | yes |
| 5 | U-s101 r64 | ≈(73.5, 87.5) | (73.4, 87.5) (`t03`) | yes |
| 5 | U-s202 r128 | ≈(73.5, 92) | (73.4, 92.0) (`t03`) | yes |
| 5 | R-s202 r128 | ≈(67.6, 90.2) | (67.6, 90.2) (`t03`) | yes |
| 6 | R-s101 r128 (J_H, J_U) | ≈(76.5, 96.1) | (76.6, 96.1) (`t03`) | yes |
| 6 | U-s101 r96 | ≈(69, 96.9) | (68.8, 96.9) (`t03`) | yes |
| 6 | H-s202 r128 | ≈(68, 90.2) | (68.0, 90.2) (`t03`) | yes |
| 6 | R-s202 r128 | ≈(67.6, 91.8) | (67.6, 91.8) (`t03`) | yes |
| 7 | control gain R-s101, upper end | ≈+28.3, ≈37 | +28.3 [+19.9, +36.9] (`t09`) | yes |
| 7 | control gain H-s202 | ≈+24.8 | +24.8 (`t09`) | yes |
| 7 | J_U gain H-s101 | ≈+13.3 | +13.3 (`t09`) | yes |
| 7 | J_R gain R-s101, lower end | ≈+9.4, ≈0.4 | +9.4 [+0.4, +18.8] (`t09`) | yes |
| 8 | hard ΔJ_H, 4 draws | ≈1.0, 2.0, 0.6, 2.1 | +1.0, +2.0, +0.6, +2.1 (`t04`) | yes |
| 8 | hard-4101 ΔJ_R | ≈−4.0 | −4.0 (`t04`) | yes |
| 8 | related-4102 ΔJ_U | ≈−4.3 | −4.3 (`t04`) | yes |
| 8 | dashed none_rerun ΔJ_H / ΔJ_R / ΔJ_U | ≈3.1 / −2.0 / −1.4 | +3.1 / −2.0 / −1.4 (`t04`) | yes |
| 9 | dashed none_rerun ΔJ_H / ΔJ_R / ΔJ_U | ≈2.5 / −0.9 / 1.2 | +2.5 / −0.9 / +1.2 (`t04`) | yes |
| 9 | unrelated-4102 ΔJ_H | ≈+2.7 | +2.7 (`t04`) | yes |
| 9 | related-4104 ΔJ_U | ≈−2.3 | −2.3 (`t04`) | yes |
| 9 | momentum-only ΔJ_H (3 crosses) | ≈0.6 | +0.6 (`t04`) | yes |
| 10 | sealed R-s101 | ≈81.9 → 84.2 | 81.9 → 84.2 (`t05`) | yes |
| 10 | sealed R-s202 | ≈74.8 → 76.9 | 74.8 → 76.9 (`t05`) | yes |
| 10 | stress H-s101 | ≈64.1 → 67.3 | 64.1 → 67.3 (`t05`) | yes |
| 11 | sealed H-s101 recovered / wrong / capped | ≈21 / 32 / 47 % | 20.9 / 31.7 / 47.4 % (93, 141, 211 of 445; `t05`) | yes |
| 11 | sealed R-s202 recovered / wrong | ≈16 / 19 % | 16.2 / 19.3 % (87, 104 of 538; `t05`) | yes |
| 11 | stress U-s202 recovered / wrong | ≈15 / 30 % | 15.3 / 30.2 % (41, 81 of 268; `t05`) | yes |
| 12 | sealed R − U seed 101 | ≈+4.2 → +3.6 | +4.2 → +3.6 (`t06`) | yes |
| 12 | sealed R − U seed 202 | ≈−4.4 → −3.9 | −4.4 → −3.9 (`t06`) | yes |
| 12 | stress R − H seed 101 | ≈+5.4 → +5.0 | +5.4 → +5.0 (`t06`) | yes |
| 13 | Harder base / R-s101 r128 | ≈52.5 / 84 | 52.5 / 84.2 (`t07`) | yes |
| 13 | SuperHard base | ≈26 | 25.8 (`t07`) | yes |
| 13 | SuperHard R-s101 r64 | ≈53.5 | 53.5 (`t07`) | yes |
| 14 | All, R − U seed 101 | ≈+4.2, ≈1.5 to 6.8 | +4.2 [+1.5, +6.8] (`t08`) | yes |
| 14 | Harder, R − U seed 202 | ≈−3.2, upper ≈+0.3 | −3.2 [−6.7, +0.3] (`t08`) | yes |
| 14 | SuperHard, U − H seed 202 | ≈+6.2, upper ≈11.1 | +6.2 [+1.5, +11.1] (`t08`) | yes |

### R2.6 Remaining items (all optional)

1. **Figure height at normal width.** The stacked layout makes 9 of 14 figures 1128–1182 px tall at 1024 px. Those
   are Figures 3–9, 13 and 14. A laptop viewport is about 1013 px tall, so the reader scrolls to reach the caption.
   To keep the phone layout and shorten the desktop view, embed each figure as
   `<img src="figures/v3/….png" width="600" alt="…">` in `REPORT.md`. GitHub keeps the `width` attribute, and
   `max-width:100%` still applies on a phone. At 600 px the tallest figure is 907 px and tick text is 16.8 px.
2. **f12 (Figure 7) panel titles.** The "J" of "J_U (32 problems)" and "J_R (28 problems)" touches the y-axis line.
   Add a small left pad to those titles (for example `ax.set_title(..., loc="left", x=0.01)`).
3. **File names.** The PNG names keep the old numbers (`f12_control.png` is Figure 7; `f07a`/`f07b` are Figures 8
   and 9). Readers do not see these names. Rename only if the mismatch confuses maintainers.

### R2.7 Screenshots

All in
`/private/tmp/claude-501/-Users-vineethbhat-Desktop-Courses-deep-rl-project-qwen-interference-study-slurm-agentic-engineering/f2158047-8407-4459-95fd-ad10c84c7e2c/scratchpad/render2/shots/`
(scratch directory; not kept in the repo):

- `fig1_w1024.png` … `fig14_w1024.png`: window 1024 × 1700 (`figN` = Figure N);
- `fig1_w420.png` … `fig14_w420.png`: 420 × 1700 viewport;
- `tab51_w420.png`, `tab55_w420.png`, `tab56_w420.png`: the three tables at 420 px.

The per-figure pages are in the sibling directory `slices/`, and the full page is `page.html`.

---

## Render 1: first figures (clone `5b211c7`, superseded by Render 2)


Purpose: check that every figure of report v3 reads correctly where it sits in the rendered report, at a normal
reading width and at a phone width. Result: all 13 figures pass at normal width. At 420 px every figure is complete
and uncut, but all figure text shrinks to 7.5–8.1 px; read it by tapping the figure (GitHub links each image to the
full-size PNG). Plotted values agree with the tables in every spot check (39 of 39 rows). Section 6 lists seven
small fixes and one optional one; none changes a number.

Date: 2026-11-11 (EST). Inspector: agent "render".

### R1.1 What was rendered

| Item | Value |
|---|---|
| Source | clean clone of `git@github.com:FlightVin/qwen-interference-study.git`, branch `main` |
| Clone SHA | `5b211c747b30715c8ac062b1e277c183161b50d9` ("Report v3: audited rewrite under A26") |
| Markdown renderer | GitHub API `POST /markdown`, `mode=markdown`, `context=FlightVin/qwen-interference-study` (the mode GitHub uses for a `.md` file in the repo view) |
| Second render | `mode=gfm` (the mode for issues and comments), used for one comparison shot only |
| Page CSS | system font 16 px, line-height 1.5, `.markdown-body{max-width:830px;padding:24px}`, `img{max-width:100%}`, tables `display:block; overflow:auto`, 1 px cell borders |
| Browser | headless Google Chrome (`--headless=new`), device scale factor 1 |
| Normal width | window 1024 px; content column 782 px |
| Narrow width | 420 px viewport (iframe of width 420, see note); content column 372 px |

Two deviations from the planned method:

1. **Per-figure pages instead of `#figN` fragments.** Headless Chrome returned a blank image for every scrolled
   viewport (a `#fig1` shot was 5.6 KB of white, with and without a scroll-on-load script). So each figure was
   rendered on its own page: the paragraph before it, the figure, and its caption, cut from the same rendered HTML with
   the same CSS. The layout of these three blocks is identical to the full page.
2. **420 px through an iframe.** Chrome on macOS will not open a window narrower than 500 px (`innerWidth` measured
   500 for `--window-size=420,…`). Each narrow page was therefore loaded into a 420 × 1000 px iframe and the shot
   cropped to that iframe.

### R1.2 Figure order in the report

| Slot in page | File | Caption number | Section |
|---|---|---|---|
| 1 | `f01_endpoints.png` | Figure 1 | 5.1 |
| 2 | `f02_contrasts_test.png` | Figure 2 | 5.1 |
| 3 | `f03_dev_rounds.png` | Figure 3 | 5.2 |
| 4 | `f04_dev_hard_attempts.png` | Figure 4 | 5.2 |
| 5 | `f05_traj_JH_JR.png` | Figure 5 | 5.2 |
| 6 | `f06_traj_JH_JU.png` | Figure 6 | 5.2 |
| 7 | `f12_control.png` | **Figure 12** | 5.3 |
| 8 | `f07_delta_j.png` | Figure 7 | 5.4.4 |
| 9 | `f08_length_pass1.png` | Figure 8 | 5.5 |
| 10 | `f08b_length_capped.png` | Figure 8b | 5.5 |
| 11 | `f09_length_contrasts.png` | Figure 9 | 5.5 |
| 12 | `f10_stress_tiers.png` | Figure 10 | 5.6 |
| 13 | `f11_stress_contrasts.png` | Figure 11 | 5.6 |

Figure 12 appears between Figures 6 and 7. A reader meets "Figure 12" before Figure 7 (fix 1).

### R1.3 Rendered text size

Every figure uses 10 pt text (titles 12 pt) at 200 dpi (`build_v3.py` lines 567–579), so text is 27.8 px tall in the
PNG. The browser scales each PNG to the content column.

| File | PNG width (px) | Scale at 1024 | Text at 1024 (px) | Scale at 420 | Text at 420 (px) | Height at 1024 (px) |
|---|---|---|---|---|---|---|
| f01 | 1271 | 0.615 | 17.1 | 0.293 | 8.1 | 444 |
| f02 | 1310 | 0.597 | 16.6 | 0.284 | 7.9 | 450 |
| f03 | 1372 | 0.570 | 15.8 | 0.271 | 7.5 | 528 |
| f04, f05, f06, f07, f08b, f09, f10, f11, f12 | 1375–1376 | 0.568 | 15.8 | 0.270 | 7.5 | 392–549; f07: 1015 |
| f08 | 1357 | 0.576 | 16.0 | 0.274 | 7.6 | 428 |

"Text" is the font size (em). At 1024 every figure clears the 10 px bar. At 420 none does; 10 px there would need
about 13.5 pt text in a 6.9-inch figure.

### R1.4 Verdicts per figure and width

Pass at 1024 means: title and axis labels not clipped, all text legible at 100% zoom, no overlapping text, legend
clear of data, caption directly below and consistent with the figure. At 420 the question is whether a reader can
read the figure without zooming.

| Slot | File | Width | Verdict | Notes |
|---|---|---|---|---|
| 1 | f01 | 1024 | pass | Caption names 128 × 32, cap and interval type. Line style (solid = seed 101, dashed = seed 202) is not explained, but every row label names its seed. |
| 1 | f01 | 420 | problem (legibility) | Complete, not clipped. Tick and row labels 8 px: readable with effort. |
| 2 | f02 | 1024 | pass | |
| 2 | f02 | 420 | problem (legibility) | Complete. Labels 8 px. |
| 3 | f03 | 1024 | pass | Legend below the panels, clear of data. Caption states no intervals are drawn and gives their size. |
| 3 | f03 | 420 | problem (legibility) | Complete. Ticks, "cap 8,192 / cap 12,288" notes and legend 7.5 px: hard to read. |
| 4 | f04 | 1024 | pass | Caption says "intervals as in Figure 3", but Figure 3 draws none (fix 6). |
| 4 | f04 | 420 | problem (legibility) | As f03. |
| 5 | f05 | 1024 | pass, minor problem | Seed 202: the H and R round-128 rings overlap near (68, 90–91), together with U round 96; the three path ends are hard to tell apart. "r0" touches the bottom spine (fix 4). |
| 5 | f05 | 420 | problem (legibility) | Seed-202 end cluster unreadable; text 7.5 px. |
| 6 | f06 | 1024 | pass, minor problem | Seed 101: H and U round-128 rings overlap at (69–70, 94); seed 202: H and R rings overlap at (68, 90–92). Two-line y-label sits just above the axes but is not clipped. |
| 6 | f06 | 420 | problem (legibility) | As f05. |
| 7 | f12 | 1024 | pass, numbering problem | Figure and caption agree. Caption number 12 is out of order (fix 1). |
| 7 | f12 | 420 | problem (legibility) | Three panels; text 7.5 px, hard to read. |
| 8 | f07 | 1024 | pass, minor problem | All 6 panels clear. Figure is 1015 px tall, so figure and caption do not fit one 1013 px screen (fix 5). |
| 8 | f07 | 420 | problem (legibility) | Complete; 482 px tall. Axis labels and legend 7.5 px. |
| 9 | f08 | 1024 | pass, minor problem | R-s202 open markers (74.8 sealed, 63.2 stress) sit on the left spine (fix 3). The two panels use different x ranges; each is labelled. |
| 9 | f08 | 420 | problem (legibility) | Text 7.6 px. |
| 10 | f08b | 1024 | pass, minor problem | x label "Share of attempts capped at 12,288 (%)" can be read as a share of all attempts; the caption says the shares are of the capped attempts (fix 7). |
| 10 | f08b | 420 | problem (legibility) | "n = …" labels and legend 7.5 px. |
| 11 | f09 | 1024 | pass | |
| 11 | f09 | 420 | problem (legibility) | Text 7.5 px. |
| 12 | f10 | 1024 | pass, minor problem | Shared x label is centred on the whole figure, not under the two panels; it sits under the left panel (fix 2). |
| 12 | f10 | 420 | problem (legibility) | Text 7.5 px. |
| 13 | f11 | 1024 | pass | Only ticks 0 and 10 per panel; SuperHard U − H seed 202 upper end (+11.1) touches the right panel edge. |
| 13 | f11 | 420 | problem (legibility) | Three panels; text 7.5 px. |

Tables at 420 px (§5.1 endpoint table, §5.5 length table, §5.6 stress table): **pass**. Each table scrolls
horizontally inside the content column; the page itself does not scroll sideways and no text overlaps.

`mode=gfm` comparison: the source is hard-wrapped at about 120 characters. In `gfm` mode (issues, comments) every
source line break becomes `<br>`, so captions break mid-sentence at 420 px ("cap / 12,288"). The repo file view
(`markdown` mode) joins the lines, so the report as browsed on GitHub is not affected.

### R1.5 Value spot checks (plot against table)

Values read off the 1024 px shots, to about ±0.5 point (±1 point on the 0–100 bar scale of f08b).

| File | Quantity | Read off plot | Table value (source) | Agree |
|---|---|---|---|---|
| f01 | base pass@1 | ≈52.5 | 52.6 (`t01`) | yes |
| f01 | R-s101 r128 pass@1 | ≈82 | 81.9 (`t01`) | yes |
| f01 | H-s101 interval | ≈71.5 to 82 | [71.5, 82.2] (`t01`) | yes |
| f02 | R − U seed 101 | ≈+4.2, bar ≈2 to 6.3 | +4.2 [+2.0, +6.3] (`t02`) | yes |
| f02 | R − U seed 202 | ≈−4.4 | −4.4 (`t02`) | yes |
| f02 | U − H seed 202 | ≈+6.8, bar ≈3.6 to 10.2 | +6.8 [+3.6, +10.2] (`t02`) | yes |
| f03 | R-s101 r64 dev | ≈73.5 | 73.6 (`t03` dev) | yes |
| f03 | U-s101 r32 dev | ≈60 | 60.1 (`t03` dev) | yes |
| f03 | U-s202 r128 dev | ≈70.3 | 70.3 (`t03` dev) | yes |
| f04 | H-s101 at 16,384 attempts | ≈66.3 | 66.3 (`t03`) | yes |
| f04 | R-s101 at 6,144 (r96) | ≈74.8 | 74.8 (`t03`) | yes |
| f04 | H-s202 at 4,096 (r32) | ≈56.6 | 56.6 (`t03`) | yes |
| f05 | H-s101 r128 (J_H, J_R) | ≈(69, 89.3) | (68.8, 89.3) (`t03`) | yes |
| f05 | U-s101 r64 | ≈(73.5, 87.5) | (73.4, 87.5) (`t03`) | yes |
| f05 | U-s202 r128 | ≈(73.5, 92) | (73.4, 92.0) (`t03`) | yes |
| f06 | R-s101 r128 (J_H, J_U) | ≈(76.5, 96.1) | (76.6, 96.1) (`t03`) | yes |
| f06 | U-s101 r96 | ≈(69, 96.9) | (68.8, 96.9) (`t03`) | yes |
| f06 | H-s202 r128 | ≈(68, 90.2) | (68.0, 90.2) (`t03`) | yes |
| f12 | control gain R-s101 | ≈+28.3, bar to ≈37 | +28.3 [+19.9, +36.9] (`t09`) | yes |
| f12 | J_U gain H-s101 | ≈+13.3 | +13.3 (`t09`) | yes |
| f12 | J_R gain R-s101, lower end | ≈0.4 | +9.4 [+0.4, +18.8] (`t09`) | yes |
| f07 | round 0, hard ΔJ_H (4 draws) | ≈1.0, 2.0, 0.6, 2.1 | +1.0, +2.0, +0.6, +2.1 (`t04`) | yes |
| f07 | round 0, hard-4101 ΔJ_R | ≈−4.0 | −4.0 (`t04`) | yes |
| f07 | dashed none_rerun, r64 ΔJ_H / ΔJ_R / ΔJ_U | ≈2.5 / −0.9 / 1.2 | +2.5 / −0.9 / +1.2 (`t04`) | yes |
| f08 | sealed R-s101 | ≈81.9 → 84.2 | 81.9 → 84.2 (`t05`) | yes |
| f08 | sealed R-s202 | ≈74.8 → 76.9 | 74.8 → 76.9 (`t05`) | yes |
| f08 | stress U-s202 | ≈65.8 → 68.2 | 65.8 → 68.2 (`t05`) | yes |
| f08b | sealed H-s101 recovered / wrong / capped | ≈21 / 32 / 47 % | 20.9 / 31.7 / 47.4 % (93, 141, 211 of 445; `t05`) | yes |
| f08b | sealed R-s101 recovered / wrong | ≈21 / 12 % | 21.5 / 11.4 % (94, 50 of 437; `t05`) | yes |
| f08b | n labels, stress column | 313, 362, 406, 407, 268 | same (`t05`) | yes |
| f09 | sealed R − U seed 101 | ≈+4.2 → +3.6 | +4.2 → +3.6 (`t06`) | yes |
| f09 | sealed U − H seed 101 | ≈+0.7 → +1.3 | +0.7 → +1.3 (`t06`) | yes |
| f09 | stress R − H seed 101 | ≈+5.4 → +5.0 | +5.4 → +5.0 (`t06`) | yes |
| f10 | Harder base / R-s101 r128 | ≈52 / 84 | 52.5 / 84.2 (`t07`) | yes |
| f10 | SuperHard base | ≈26 | 25.8 (`t07`) | yes |
| f10 | Harder H-s202 r64 | ≈76 | 76.1 (`t07`) | yes |
| f11 | All, R − U seed 101 | ≈+4.2, bar ≈1.5 to 6.8 | +4.2 [+1.5, +6.8] (`t08`) | yes |
| f11 | Harder, R − U seed 202 | ≈−3.2, bar to ≈+0.3 | −3.2 [−6.7, +0.3] (`t08`) | yes |
| f11 | SuperHard, U − H seed 202 | ≈+6.2, bar to ≈11 | +6.2 [+1.5, +11.1] (`t08`) | yes |

### R1.6 Fixes needed (most important first)

1. **`REPORT.md` §5.3 and `build_v3.py` (f12): figure numbering out of order.** `f12_control.png` is captioned
   "Figure 12" but is the 7th figure in the page. Renumber so captions run 1–12 in page order (control becomes
   Figure 7; current 7–11 shift by one), or move the control figure after Figure 11. Update every "Figure N" cross
   reference and `captions.md`.
2. **f10 (`build_v3.py` line 934): shared x label off-centre.** `fig.supxlabel(..., y=0.06)` centres on the whole
   figure, so the label sits under the left panel. Centre it on the panels (for example
   `x=(axes[0].get_position().x0 + axes[1].get_position().x1) / 2`) or set the label on each axis.
3. **f08: markers on the left spine.** R-s202 open markers (74.8 sealed, 63.2 stress) sit on the y-axis line. Widen
   the left x limit by about 1 point in both panels.
4. **f05, f06: crowded path ends.** Round-128 rings overlap (f05 seed 202: H and R; f06 seed 101: H and U; f06 seed
   202: H and R). Reduce ring size, or offset the rings slightly, or label each ring with its arm letter. Move the "r0"
   annotation (`build_v3.py` line 737, `xytext=(0, -8)`) up or right so it clears the bottom spine.
5. **f07: too tall for one screen.** At 1024 px the figure is 1015 px tall; the caption falls below the first
   screen. Reduce `figsize=(6.9, 8.2)` (line 772) to about 6.9 × 6.6 in, or move the legend into the top margin.
6. **`REPORT.md` Figure 4 caption.** "Markers and intervals as in Figure 3" — Figure 3 draws no intervals. Change to
   "Markers as in Figure 3; no intervals drawn (sizes in `t03_curves`)."
7. **f08b x label.** "Share of attempts capped at 12,288 (%)" → "Share of the capped attempts (%)", to match the
   caption.
8. **Narrow screens (all figures; optional).** At 420 px all figure text is 7.5–8.1 px. Raising 10 pt to 12 pt only
   reaches 9–10 px and would crowd the 3-panel figures (f11, f12). Recommendation: keep the current size and add one
   sentence near the start of §5: "On a phone, tap a figure to open it at full size." GitHub already links each image
   to its full-size PNG.

### R1.7 Screenshots

All in
`/private/tmp/claude-501/-Users-vineethbhat-Desktop-Courses-deep-rl-project-qwen-interference-study-slurm-agentic-engineering/f2158047-8407-4459-95fd-ad10c84c7e2c/scratchpad/render/shots/`
(scratch directory; not kept in the repo):

- `fig1_w1024.png` … `fig13_w1024.png`: normal width, one per page slot (section 2 maps slot to file);
- `fig8_w1024_tall.png`: slot 8 (f07) at 1024 × 1500 so its caption is visible;
- `fig1_w420.png` … `fig13_w420.png`: 420 px viewport;
- `tab51_w420.png`, `tab55_w420.png`, `tab56_w420.png`: the three tables at 420 px;
- `fig1_w420_gfm.png`: slot 1 rendered in `gfm` mode, for the line-break comparison.

The rendered pages are in the sibling directories `slices/` (per-figure pages) and the full pages `page.html`
(`markdown` mode) and `page_gfm.html`.
