# gPRISM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: ≥3.9](https://img.shields.io/badge/python-%E2%89%A53.9-blue.svg)](https://www.python.org/downloads/)
[![Status: Manuscript in preparation](https://img.shields.io/badge/status-manuscript%20in%20preparation-orange.svg)](#)

**Probabilistic reconstruction of individual genotypes from sequencing mixtures.**

gPRISM is a unified probabilistic framework for multi-individual DNA and RNA
mixture deconvolution from next-generation sequencing data. Given aligned
pile-up data, gPRISM jointly infers

1. the optimal number of contributors (NoC) via BIC-based model selection,
2. the mixture proportion of each contributor,
3. SNP-level genotype posterior probabilities for every contributor, and
4. reconstructed contributor-specific genotypes,

within a single beta-binomial mixture model fitted by a generalised
expectation–maximisation (EM) algorithm.

The framework is described in:

> Lee S.\*, Park D.\*, Kim T.\*, Yang I.S., Park S.U., Kwon Y.-L., Park S.,
> Shin H.J., Ko A., Shin K.-J., Lee H.Y., Kang H.-C., Kim S. **Probabilistic
> reconstruction of individual genotypes from sequencing mixtures using
> gPRISM.** *(Manuscript in preparation.)*

## Installation

```bash
git clone https://github.com/sangbolee/gprism.git
cd gprism
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Requirements: Python ≥ 3.9, NumPy, SciPy, PyTorch ≥ 2.0.

### Generating the input mpileup

gPRISM takes per-sample gzip-compressed mpileup files
(`<sample>.mpileup.gz`) as input, produced upstream with `samtools`:

```bash
samtools mpileup -f reference.fa -q 20 -Q 20 sample.bam | gzip > sample.mpileup.gz
```

`samtools` (≥ 1.9 recommended) is not a Python dependency — install it via
your package manager (e.g. `brew install samtools`, `conda install -c bioconda samtools`).

## Quick start

The pipeline has three stages, each runnable independently or chained with
`gprism run`. A one-shot preparation step, `gprism build-popaf`, builds the
reference VCF that stage 1 uses as input.

### 0. Build a PoPAF reference VCF (one-time)

The `parse` stage below requires a gzip-compressed reference VCF containing
only biallelic SNPs, whose `INFO` column is reduced to
`AC[_pop]=<v>;AF[_pop]=<v>` (AC first, AF second). Use `gprism build-popaf` to
produce one from a raw gnomAD or Korea4K release:

```bash
# East-Asian allele frequencies from gnomAD
gprism build-popaf \
    --input  gnomad.genomes.v4.0.sites.vcf.gz \
    --output gnomAD.PoPAF.eas.biallelic.vcf.gz \
    --pop    eas

# Korea4K (universal AC/AF only)
gprism build-popaf \
    --input  korea4k_fixed.all.vcf.gz \
    --output Korea4K.PoPAF.biallelic.vcf.gz
```

The resulting `.vcf.gz` is the `--db-path` argument for stage 1 below.

Options:

| Flag              | Default                    | Description                                                  |
|-------------------|----------------------------|--------------------------------------------------------------|
| `--input`         | *required*                 | Input `.vcf.gz`.                                             |
| `--output`        | `PoPAF.biallelic.vcf.gz`   | Output `.vcf.gz`.                                            |
| `--pop`           | `""`                       | Population suffix (empty = universal `AC`/`AF`).             |
| `--workdir`       | `./vcf_split_temp`         | Temporary directory (removed on completion).                 |
| `--keep-workdir`  | off                        | Retain intermediate `indel.vcf` / `multiallelic.vcf` splits. |

Indels, multiallelic sites, and variants missing the requested `AC_<pop>` /
`AF_<pop>` fields are dropped; a summary of kept / skipped / indel / multi
counts is printed to stderr. See [`examples/popaf/`](examples/popaf/) for
reference input/output snippets. Any population suffix that appears in the
input VCF's `INFO` as `AC_<pop>` / `AF_<pop>` can be passed to `--pop`;
refer to the [gnomAD population documentation](https://gnomad.broadinstitute.org/help/ancestry)
for the full list of supported codes.

### 1. Parse raw mpileup

```bash
gprism parse \
    --input  /path/to/mpileup_dir \
    --output /path/to/parsed_dir \
    --sample SAMPLE01 \
    --db-path /path/to/Korea4K.PoPAF.biallelic.vcf.gz \
    --db-type korea4k        # or gnomad
```

Writes `SAMPLE01.biallelic.mpileup` — a tab-delimited file with
`CHROM / POS / REF / ALT / Depth / AC / VAF / PAF / Bases / BQ`.

### 2. Deconvolve (NoC + mixture proportions)

```bash
gprism deconvolve \
    --input  /path/to/parsed_dir \
    --sample SAMPLE01 \
    --output /path/to/results \
    --max-k  5
```

Options:

| Flag       | Default    | Description                                                                                      |
|------------|------------|--------------------------------------------------------------------------------------------------|
| `--input`  | *required* | Directory containing `<sample>.biallelic.mpileup`.                                               |
| `--output` | *required* | Output directory for `<sample>.Mixture_Results.txt`.                                             |
| `--sample` | *required* | Sample identifier.                                                                               |
| `--phi`    | `200`      | Dispersion. A float (e.g. `30` for low-coverage data) or `auto`/`depth`/`median` for median depth. |
| `--max-k`  | `5`        | Maximum number of contributors K to try (fits K = 1, 2, …, max-k).                               |
| `--n-init` | `10`       | EM restarts per K with random initial θ (multi-init to escape local minima).                     |
| `--device` | auto       | Torch device: `cpu`, `cuda`, `cuda:0`, etc. Empty = auto-detect CUDA.                            |

Writes `SAMPLE01.Mixture_Results.txt` with per-K BIC, log-likelihood, mixture
proportions, fitted centers, and weights.

### 3. Genotype reconstruction

```bash
gprism genotype \
    --input           /path/to/parsed_dir/SAMPLE01.biallelic.mpileup \
    --mixture-results /path/to/results/SAMPLE01.Mixture_Results.txt \
    --sample          SAMPLE01 \
    --output          /path/to/results \
    --mode            posterior   # or 'hard'
```

Options:

| Flag                | Default      | Description                                                                              |
|---------------------|--------------|------------------------------------------------------------------------------------------|
| `--input`           | *required*   | Path to `<sample>.biallelic.mpileup`.                                                    |
| `--mixture-results` | *required*   | Path to `<sample>.Mixture_Results.txt` from stage 2.                                     |
| `--sample`          | *required*   | Sample identifier.                                                                       |
| `--output`          | *required*   | Output directory for `<sample>.Membership.txt`.                                          |
| `--mode`            | `posterior`  | `posterior` (probability triplet per locus) or `hard` (discrete MAP call + confidence).  |
| `--phi`             | `200`        | Dispersion used for posterior computation. Float or `auto`/`depth`/`median`. **Should match the value used in stage 2.** |

Writes `SAMPLE01.Membership.txt`:

* `--mode posterior` — per-locus `(ref, hetero, homo)` probability triplet for
  each contributor.
* `--mode hard`      — discrete MAP call (`ref`, `alt_hetero`, `alt_homo`,
  `alt_ambi`, or `ambiguous`) plus the confidence score.

### Full pipeline in one go

```bash
gprism run \
    --input   /path/to/mpileup_dir \
    --output  /path/to/results \
    --sample  SAMPLE01 \
    --db-path /path/to/Korea4K.PoPAF.biallelic.vcf.gz \
    --db-type korea4k \
    --phi     200 \
    --max-k   5 \
    --mode    posterior
```

Options:

| Flag        | Default      | Description                                                                                      |
|-------------|--------------|--------------------------------------------------------------------------------------------------|
| `--input`   | *required*   | Directory containing `<sample>.mpileup.gz`.                                                      |
| `--output`  | *required*   | Output directory for all intermediate and final files.                                           |
| `--sample`  | *required*   | Sample identifier.                                                                               |
| `--db-path` | *required*   | PoPAF reference `.vcf.gz` produced by `gprism build-popaf`.                                      |
| `--db-type` | `korea4k`    | `korea4k` or `gnomad` (controls chromosome-prefix normalization — see stage 1).                  |
| `--phi`     | `200`        | Dispersion. Float (e.g. `30` for low-coverage data) or `auto`/`depth`/`median` for median depth. |
| `--max-k`   | `5`          | Maximum number of contributors to try.                                                           |
| `--mode`    | `posterior`  | Genotype output mode: `posterior` or `hard`.                                                     |
| `--device`  | auto         | Torch device: `cpu`, `cuda`, `cuda:0`, etc.                                                      |

### Dispersion parameter `--phi`

* `--phi 200` (default) — fixed concentration, backward-compatible with the
  prototype implementation.
* `--phi auto` / `--phi depth` — sets `phi` to the median read depth at run
  time, matching the manuscript definition.

### Depth filtering (adaptive)

gPRISM automatically adapts the minimum retained depth based on the resolved
`phi` value, so the same command works across high- and low-coverage datasets
without manual tuning:

| Resolved `phi` | Effective `min_depth` |
|----------------|-----------------------|
| `phi ≥ 100`    | `50` (manuscript default) |
| `phi < 100`    | `max(phi / 2, 10)` |

For example, `--phi 30` yields `min_depth = 15`, and `--phi 200` keeps the
legacy `50`. Sites below the threshold are dropped before EM runs.

> ⚠️ **Low-phi warning.** When the resolved `phi` is ≤ 20, gPRISM emits a
> `UserWarning: too low depth for the algorithm; results may be unreliable.`
> to stderr. In that regime both the beta-binomial fit and the genotype
> posteriors become noisy. Consider aggregating replicates, increasing
> coverage, or interpreting results with care.

### Resolution floor (EM internals)

Inside the EM loop, the mixture-proportion vector `a_vec` is prevented from
collapsing onto the simplex boundary by a *resolution floor*:

```
a_vec = resolution + (1 − K × resolution) × softmax(θ)
resolution = min(4.04 / mean_depth, 0.05)
```

The `4.04 / mean_depth` term is the inverse-model fit reported in
Supplementary Fig. 2 of the manuscript. The hard cap at `0.05` prevents
`1 − K × resolution` from going non-positive at very shallow depths (e.g.
`mean_depth = 10, K = 3` would yield `1 − 3 × 0.404 < 0` without the cap).
Users normally do not need to tune this — it is exposed through
`GPRISMConfig.resolution_constant` and `resolution_ceiling` in the Python
API.

## Output formats

All output files are plain tab-separated text with a single header row.

### `<sample>.biallelic.mpileup` (stage 1)

| Column | Meaning |
|--------|---------|
| `CHROM` | Chromosome |
| `POS` | 1-based genomic position |
| `REF` / `ALT` | Reference / alternate alleles from the PoPAF VCF |
| `Depth` | Total read depth at the site |
| `AC` | Alt-allele read count |
| `VAF` | Variant allele fraction (`AC / Depth`) |
| `PAF` | Population allele frequency carried from the PoPAF VCF |
| `Bases` | Raw pileup base string (for traceability) |
| `BQ` | Raw base-quality string |

### `<sample>.Mixture_Results.txt` (stage 2)

One row per tested `K`. Column set differs between `K = 1` and `K ≥ 2`:

| Column | `K = 1` | `K ≥ 2` |
|--------|---------|---------|
| `Sample` | sample id | sample id |
| `NoC` | `1` | `2, 3, …, max-k` |
| `MixtureProp` | `1.0` | comma-separated mixture proportions, e.g. `0.72,0.28` |
| `Center` | *(empty)* | comma-separated expected centers for every genotype combination (length `3^K`) |
| `Weight` | *(empty)* | comma-separated soft-responsibility weights per combination |
| `BIC` | model BIC | model BIC |
| `LogL` | *(empty)* | total log-likelihood |

The optimal K is simply `argmin_K BIC`; it is re-derived on the fly by
downstream steps (`gprism genotype` / `gprism run`).

### `<sample>.Membership.txt` (stage 3)

`--mode posterior` — per-locus posterior probability triplet per contributor:

| Column                        | Meaning                                              |
|-------------------------------|------------------------------------------------------|
| `Sample`                      | sample id                                            |
| `Chr`                         | chromosome                                           |
| `Position`                    | 1-based position                                     |
| `Contributor#k_ref_prob`      | posterior P(contributor k is homozygous-ref)         |
| `Contributor#k_hetero_prob`   | posterior P(contributor k is heterozygous)           |
| `Contributor#k_homo_prob`     | posterior P(contributor k is homozygous-alt)         |

Columns `Contributor#1_…`, `Contributor#2_…`, … repeat for each of the `K`
contributors. The three per-contributor probabilities sum to 1.

`--mode hard` — per-locus discrete MAP call per contributor:

| Column                    | Meaning                                                                 |
|---------------------------|-------------------------------------------------------------------------|
| `Sample`                  | sample id                                                               |
| `Chr`                     | chromosome                                                              |
| `Position`                | 1-based position                                                        |
| `Contributor#k_GT`        | one of `ref`, `alt_hetero`, `alt_homo`, `alt_ambi`, `ambiguous`         |
| `Contributor#k_Prob`      | posterior mass of the chosen call (`.5f` float), or `.` if ambiguous     |

Classification thresholds (`ref_alt_threshold = 4.0`, `homo_hetero_threshold
= 9.0`) are exposed via `GPRISMConfig` in the Python API.

## Project layout

```
gprism/
├── gprism/
│   ├── cli.py              # argparse entry point
│   ├── config.py           # GPRISMConfig (all tunable parameters)
│   ├── io/
│   │   ├── parser.py       # unified Korea4K / gnomAD mpileup parser
│   │   ├── popaf.py        # biallelic-SNP PoPAF reference VCF builder
│   │   └── writer.py       # Mixture_Results / Membership writers
│   ├── model/
│   │   ├── likelihood.py   # beta-binomial log-likelihood
│   │   ├── em.py           # generalised EM (PyTorch)
│   │   └── selection.py    # BIC model selection over K = 1..max_k
│   ├── genotype/
│   │   ├── posterior.py    # posterior probability computation
│   │   └── reconstruction.py  # hard / posterior genotype writer
│   └── utils/math.py       # shared beta-binomial / HWE / combo utilities
├── examples/
│   └── popaf/              # reference VCF snippets for build-popaf
├── tests/                  # pytest suite (unit + smoke tests)
├── pyproject.toml
├── LICENSE                 # MIT
└── README.md
```

## Citation

If you use gPRISM in your research, please cite the manuscript above. A full
citation block will be added here upon publication.

## License

Released under the [MIT License](LICENSE).
