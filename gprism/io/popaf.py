"""Build a biallelic-SNP population-allele-frequency (PoPAF) reference VCF.

Filters a raw population VCF (Korea4K, gnomAD, or any VCF with ``AC``/``AF``
in its ``INFO`` field) down to **biallelic SNPs only**, and rewrites the
``INFO`` column to exactly ``AC[_pop]=<val>;AF[_pop]=<val>`` (AC first,
AF second).

The output VCF is the reference database consumed by
:func:`gprism.io.parser.load_snp_database`, which reads the population
allele frequency via::

    popaf = float(fields[7].split(";")[1].split("=")[1])

so the output **must** contain exactly two ``;``-separated INFO entries,
in the order ``AC`` → ``AF``. This module enforces that contract and
self-verifies every run.

Algorithm (mirrors the original Korea4K preparation notebook):

1. Split header vs. data.
2. Split indel vs. SNP records by REF/ALT length.
3. Split multiallelic vs. biallelic SNPs via a :class:`~collections.Counter`
   on ``(CHROM, POS)``.
4. Rewrite the INFO column of each biallelic SNP to the AC/AF pair of
   interest and write header + body to a gzip-compressed output VCF.

Stdlib-only — no ``bcftools`` / ``pysam`` dependency.
"""

from __future__ import annotations

import gzip
import os
import shutil
from collections import Counter
from typing import Optional


def pick(info: str, key: str) -> Optional[str]:
    """Return the value for ``key=`` in a ``;``-separated INFO string, or ``None``."""
    prefix = key + "="
    for kv in info.split(";"):
        if kv.startswith(prefix):
            return kv[len(prefix):]
    return None


def parse_popaf_vcf(
    input_path: str,
    output_path: str,
    pop: str = "",
    workdir: str = "./vcf_split_temp",
    cleanup: bool = True,
) -> dict:
    """Filter ``input_path`` down to biallelic SNPs with a trimmed INFO column.

    Parameters
    ----------
    input_path : str
        Gzip-compressed input VCF (``.vcf.gz``).
    output_path : str
        Output path; will be gzip-compressed.
    pop : str
        Population code. ``""`` (default) selects universal ``AC``/``AF``;
        e.g. ``"eas"`` selects ``AC_eas``/``AF_eas``.
    workdir : str
        Temporary directory for intermediate split files.
    cleanup : bool
        Remove ``workdir`` on completion.

    Returns
    -------
    dict
        Summary with keys ``kept``, ``skipped``, ``indels``, ``multi``,
        ``verified``, ``ac_key``, ``af_key``, ``output``.
    """
    in_vcf = os.path.abspath(input_path)
    out_gz = os.path.abspath(output_path)
    workdir = os.path.abspath(workdir)

    ac_key = "AC" if pop == "" else f"AC_{pop}"
    af_key = "AF" if pop == "" else f"AF_{pop}"

    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.dirname(out_gz) or ".", exist_ok=True)

    header_path       = os.path.join(workdir, "header.vcf")
    data_path         = os.path.join(workdir, "data.vcf")
    indel_path        = os.path.join(workdir, "indel.vcf")
    snp_only_path     = os.path.join(workdir, "snp_only.vcf")
    multiallelic_path = os.path.join(workdir, "multiallelic.vcf")
    biallelic_path    = os.path.join(workdir, "biallelic.vcf")

    # 1) HEADER / DATA split
    with gzip.open(in_vcf, "rt") as infile, \
         open(header_path, "w") as hdr_out, \
         open(data_path, "w") as dat_out:
        for line in infile:
            (hdr_out if line.startswith("#") else dat_out).write(line)

    # 2) INDEL vs SNP split
    indel_records: list = []
    snp_records: list = []
    with open(data_path) as dat_in:
        for line in dat_in:
            cols = line.rstrip("\n").split("\t")
            ref, alt = cols[3], cols[4]
            if len(ref) > 1 or len(alt) > 1:
                indel_records.append(line)
            else:
                snp_records.append(line)
    with open(indel_path, "w") as f:
        f.writelines(indel_records)
    with open(snp_only_path, "w") as f:
        f.writelines(snp_records)

    # 3) MULTIALLELIC vs BIALLELIC split + INFO rewrite
    pos_counter = Counter(
        (line.split("\t")[0], line.split("\t")[1]) for line in snp_records
    )
    multis = {pos for pos, cnt in pos_counter.items() if cnt > 1}

    missing_count = 0
    kept_count = 0
    multi_count = sum(pos_counter[p] for p in multis)

    with open(multiallelic_path, "w") as multi_out, \
         open(biallelic_path, "w") as bi_out:
        for line in snp_records:
            cols = line.rstrip("\n").split("\t")
            chrom, pos = cols[0], cols[1]
            if (chrom, pos) in multis:
                multi_out.write(line)
                continue

            id_, ref, alt, qual, filt, info = (
                cols[2], cols[3], cols[4], cols[5], cols[6], cols[7]
            )

            ac = pick(info, ac_key)
            af = pick(info, af_key)
            if ac is None or af is None:
                missing_count += 1
                continue
            try:
                float(af)
            except ValueError:
                missing_count += 1
                continue

            new_info = f"{ac_key}={ac};{af_key}={af}"
            bi_out.write(
                f"{chrom}\t{pos}\t{id_}\t{ref}\t{alt}\t{qual}\t{filt}\t{new_info}\n"
            )
            kept_count += 1

    # 4) HEADER + biallelic body -> gzip
    with open(header_path) as h, \
         open(biallelic_path) as b, \
         gzip.open(out_gz, "wt") as o:
        for line in h:
            o.write(line)
        for line in b:
            o.write(line)

    # 5) Cleanup
    if cleanup:
        shutil.rmtree(workdir)

    # 6) End-of-run self-verification — enforces the gPRISM parser contract
    verified = _verify_output(out_gz, ac_key, af_key)

    return {
        "kept": kept_count,
        "skipped": missing_count,
        "indels": len(indel_records),
        "multi": multi_count,
        "verified": verified,
        "ac_key": ac_key,
        "af_key": af_key,
        "output": out_gz,
    }


def _verify_output(out_gz: str, ac_key: str, af_key: str) -> int:
    """Re-read the produced VCF and assert it matches the gPRISM contract."""
    seen: set = set()
    count = 0
    with gzip.open(out_gz, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            assert len(fields[3]) == 1 and len(fields[4]) == 1, \
                f"non-SNP slipped through: {fields[:5]}"
            parts = fields[7].split(";")
            assert len(parts) == 2, f"INFO must have exactly 2 fields: {fields[7]}"
            assert parts[0].startswith(ac_key + "="), \
                f"1st INFO field not {ac_key}=: {parts[0]}"
            assert parts[1].startswith(af_key + "="), \
                f"2nd INFO field not {af_key}=: {parts[1]}"
            float(parts[1].split("=")[1])  # gPRISM's exact expression
            key = (fields[0], fields[1])
            assert key not in seen, f"duplicate (CHROM, POS): {key}"
            seen.add(key)
            count += 1
    return count
