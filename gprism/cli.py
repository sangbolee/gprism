"""Command-line interface for gPRISM."""

import argparse
import sys
from pathlib import Path

from gprism.config import GPRISMConfig, DEFAULT_PHI


def _parse_phi(value):
    """Interpret the --phi CLI argument.

    Accepts a float, ``"auto"``, or ``"depth"``. The last two return
    *None*, which downstream code interprets as "use median read depth",
    matching the manuscript definition.
    """
    if value is None:
        return DEFAULT_PHI
    if isinstance(value, str) and value.lower() in ("auto", "depth", "median"):
        return None
    return float(value)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="gprism",
        description="gPRISM: Probabilistic reconstruction of individual "
                    "genotypes from sequencing mixtures",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- build-popaf ---
    p_popaf = sub.add_parser(
        "build-popaf",
        help="Build a biallelic-SNP PoPAF reference VCF from a raw population VCF",
    )
    p_popaf.add_argument("--input", required=True,
                         help="Input .vcf.gz (e.g. gnomAD or Korea4K)")
    p_popaf.add_argument("--output", default="PoPAF.biallelic.vcf.gz",
                         help="Output .vcf.gz (default: %(default)s)")
    p_popaf.add_argument("--pop", default="",
                         help='Population code, e.g. "eas", "afr". Empty means universal AC/AF.')
    p_popaf.add_argument("--workdir", default="./vcf_split_temp",
                         help="Temporary directory (default: %(default)s)")
    p_popaf.add_argument("--keep-workdir", action="store_true",
                         help="Retain the intermediate split files for inspection")

    # --- parse ---
    p_parse = sub.add_parser("parse", help="Parse raw mpileup against SNP database")
    p_parse.add_argument("--input", required=True, help="Directory with <sample>.mpileup.gz")
    p_parse.add_argument("--output", required=True, help="Output directory")
    p_parse.add_argument("--sample", required=True, help="Sample name")
    p_parse.add_argument("--db-path", required=True, help="Path to reference VCF.gz")
    p_parse.add_argument("--db-type", default="korea4k", choices=["korea4k", "gnomad"])

    # --- deconvolve ---
    p_deconv = sub.add_parser("deconvolve", help="Run EM model selection (K=1..max_k)")
    p_deconv.add_argument("--input", required=True, help="Directory with parsed .biallelic.mpileup")
    p_deconv.add_argument("--sample", required=True, help="Sample name")
    p_deconv.add_argument("--output", required=True, help="Output directory")
    p_deconv.add_argument(
        "--phi", default=str(DEFAULT_PHI),
        help=f"Dispersion: float value (default {DEFAULT_PHI}) or 'auto'/'depth' "
             "for median read depth (matches manuscript definition)",
    )
    p_deconv.add_argument("--max-k", type=int, default=5)
    p_deconv.add_argument("--n-init", type=int, default=10)
    p_deconv.add_argument("--device", default=None)

    # --- genotype ---
    p_geno = sub.add_parser("genotype", help="Reconstruct contributor genotypes")
    p_geno.add_argument("--input", required=True, help="Path to .biallelic.mpileup")
    p_geno.add_argument("--mixture-results", required=True, help="Path to .Mixture_Results.txt")
    p_geno.add_argument("--sample", required=True, help="Sample name")
    p_geno.add_argument("--output", required=True, help="Output directory")
    p_geno.add_argument("--mode", default="posterior", choices=["hard", "posterior"])
    p_geno.add_argument(
        "--phi", default=str(DEFAULT_PHI),
        help=f"Dispersion: float (default {DEFAULT_PHI}) or 'auto'/'depth' for median depth",
    )

    # --- run (full pipeline) ---
    p_run = sub.add_parser("run", help="Full pipeline: parse -> deconvolve -> genotype")
    p_run.add_argument("--input", required=True, help="Directory with <sample>.mpileup.gz")
    p_run.add_argument("--output", required=True, help="Output directory")
    p_run.add_argument("--sample", required=True, help="Sample name")
    p_run.add_argument("--db-path", required=True, help="Path to reference VCF.gz")
    p_run.add_argument("--db-type", default="korea4k", choices=["korea4k", "gnomad"])
    p_run.add_argument(
        "--phi", default=str(DEFAULT_PHI),
        help=f"Dispersion: float (default {DEFAULT_PHI}) or 'auto'/'depth' for median depth",
    )
    p_run.add_argument("--max-k", type=int, default=5)
    p_run.add_argument("--mode", default="posterior", choices=["hard", "posterior"])
    p_run.add_argument("--device", default=None)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "build-popaf":
        from gprism.io.popaf import parse_popaf_vcf
        result = parse_popaf_vcf(
            input_path=args.input,
            output_path=args.output,
            pop=args.pop,
            workdir=args.workdir,
            cleanup=not args.keep_workdir,
        )
        print(
            f"[gprism build-popaf] pop={args.pop!r} -> keys "
            f"({result['ac_key']}, {result['af_key']})\n"
            f"  kept:     {result['kept']} biallelic SNPs\n"
            f"  skipped:  {result['skipped']} (missing/invalid "
            f"{result['ac_key']} or {result['af_key']})\n"
            f"  indels:   {result['indels']}\n"
            f"  multi:    {result['multi']}\n"
            f"  verified: {result['verified']} lines in {result['output']}",
            file=sys.stderr,
        )

    elif args.command == "parse":
        from gprism.io.parser import parse_mpileup
        parse_mpileup(args.input, args.output, args.sample,
                      args.db_path, args.db_type)

    elif args.command == "deconvolve":
        from gprism.model.selection import select_optimal_k
        from gprism.io.writer import write_mixture_results

        config = GPRISMConfig(
            phi=_parse_phi(args.phi), max_k=args.max_k,
            n_init=args.n_init, device=args.device,
        )
        data_path = str(Path(args.input) / f"{args.sample}.biallelic.mpileup")
        results = select_optimal_k(data_path, args.sample, config)

        out = str(Path(args.output) / f"{args.sample}.Mixture_Results.txt")
        write_mixture_results(out, args.sample, results)

    elif args.command == "genotype":
        from gprism.genotype.reconstruction import reconstruct_genotypes

        config = GPRISMConfig(phi=_parse_phi(args.phi))
        data_path = str(Path(args.input))
        out = str(Path(args.output) / f"{args.sample}.Membership.txt")
        reconstruct_genotypes(data_path, args.mixture_results, args.sample,
                              out, mode=args.mode, config=config)

    elif args.command == "run":
        from gprism.io.parser import parse_mpileup
        from gprism.model.selection import select_optimal_k
        from gprism.io.writer import write_mixture_results
        from gprism.genotype.reconstruction import reconstruct_genotypes

        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        config = GPRISMConfig(
            phi=_parse_phi(args.phi), max_k=args.max_k, device=args.device,
        )

        print(f"=== Step 1/3: Parsing {args.sample} ===")
        parse_mpileup(args.input, str(out_dir), args.sample,
                      args.db_path, args.db_type)

        print("\n=== Step 2/3: Deconvolution ===")
        data_path = str(out_dir / f"{args.sample}.biallelic.mpileup")
        results = select_optimal_k(data_path, args.sample, config)
        mix_out = str(out_dir / f"{args.sample}.Mixture_Results.txt")
        write_mixture_results(mix_out, args.sample, results)

        print("\n=== Step 3/3: Genotype reconstruction ===")
        mem_out = str(out_dir / f"{args.sample}.Membership.txt")
        reconstruct_genotypes(data_path, mix_out, args.sample,
                              mem_out, mode=args.mode, config=config)

        print(f"\n=== Done: {args.sample} ===")


if __name__ == "__main__":
    main()
