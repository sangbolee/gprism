"""Unified mpileup parser supporting Korea4K and gnomAD reference databases."""

import gzip
import time
from pathlib import Path
from typing import Optional, Tuple

from gprism.config import DEFAULT_POPAF_RANGE


def load_snp_database(db_path: str, db_type: str = "korea4k",
                      popaf_range: Optional[Tuple[float, float]] = None):
    """Load an SNP reference database (Korea4K or gnomAD) into a dictionary.

    Parameters
    ----------
    db_path : str
        Path to the gzip-compressed VCF file.
    db_type : str
        ``"korea4k"`` or ``"gnomad"`` (or any other, same filter applies).
    popaf_range : tuple of float, optional
        If provided, only retain SNPs whose PopAF falls strictly within
        ``(low, high)``. Defaults to :data:`DEFAULT_POPAF_RANGE` when *None*,
        irrespective of ``db_type``.

    Returns
    -------
    dict[(str, str), (str, str, float)]
        ``{(chrom, pos): (ref, alt, PopAF)}``.
    """
    snp_dict = {}

    if popaf_range is None:
        popaf_range = DEFAULT_POPAF_RANGE

    with gzip.open(db_path, mode="rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            chrom = fields[0]
            pos = fields[1]
            ref = fields[3]
            alt = fields[4]
            popaf = float(fields[7].split(";")[1].split("=")[1])

            if popaf_range and not (popaf_range[0] < popaf < popaf_range[1]):
                continue

            # Normalise chromosome naming for gnomAD
            if db_type == "gnomad" and not chrom.startswith("chr"):
                chrom = "chr" + chrom

            snp_dict[(chrom, pos)] = (ref, alt, popaf)

    return snp_dict


def parse_mpileup(input_path: str, output_path: str, sample_name: str,
                  db_path: str, db_type: str = "korea4k",
                  popaf_range: Optional[Tuple[float, float]] = None):
    """Parse a raw mpileup file against an SNP reference database.

    Reads ``<input_path>/<sample_name>.mpileup.gz``, filters against the
    reference database, counts alleles, and writes a ``.biallelic.mpileup``
    TSV.

    Parameters
    ----------
    input_path : str
        Directory containing ``<sample_name>.mpileup.gz``.
    output_path : str
        Directory for the output ``.biallelic.mpileup`` file.
    sample_name : str
        Sample identifier.
    db_path : str
        Path to the gzip-compressed reference VCF.
    db_type : str
        ``"korea4k"`` or ``"gnomad"``.
    popaf_range : tuple of float, optional
        PopAF filter applied when loading the database.

    Returns
    -------
    int
        Number of variants written.
    """
    snp_db = load_snp_database(db_path, db_type, popaf_range)

    mpileup_file = str(Path(input_path) / f"{sample_name}.mpileup.gz")
    out_file = str(Path(output_path) / f"{sample_name}.biallelic.mpileup")

    start = time.time()
    num = 0

    with open(out_file, mode="w") as out:
        out.write("#CHROM\tPOS\tREF\tALT\tDepth\tAC\tVAF\tPAF\tBases\tBQ\n")

        with gzip.open(mpileup_file, mode="rt") as fh:
            for line in fh:
                fields = line.strip().split("\t")
                chrom, pos = fields[0], fields[1]

                # Handle chr prefix for gnomAD matching
                if db_type == "gnomad" and not chrom.startswith("chr"):
                    contig = ("chr" + chrom, pos)
                else:
                    contig = (chrom, pos)

                if contig not in snp_db:
                    continue

                ref, alt, popaf = snp_db[contig]
                bases = fields[4]
                bq = fields[5]

                valid_bases = [
                    c for c in bases
                    if c.upper() in ("A", "T", "G", "C", ".", ",")
                ]
                depth = len(valid_bases)
                if depth == 0:
                    continue

                alt_count = sum(1 for c in valid_bases if c.lower() == alt.lower())
                num += 1

                out.write(
                    f"{contig[0]}\t{pos}\t{ref}\t{alt}\t{depth}\t{alt_count}\t"
                    f"{alt_count / float(depth):.6f}\t{popaf}\t{bases}\t{bq}\n"
                )

    elapsed = (time.time() - start) / 60
    print(f"{num} variants written in {elapsed:.2f} min")
    return num
