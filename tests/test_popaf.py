"""Tests for :mod:`gprism.io.popaf`."""

from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path

from gprism.io.popaf import parse_popaf_vcf, pick


TOY_VCF = [
    "##fileformat=VCFv4.2\n",
    "##contig=<ID=chr1,length=100>\n",
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n",
    # indel (dropped)
    "chr1\t100\t.\tAT\tA\t.\tPASS\tAC=1;AF=0.0001;AC_eas=0;AF_eas=0\n",
    # biallelic SNP (kept)
    "chr1\t200\trs1\tT\tC\t.\tPASS\tAC=3;AF=0.003;AC_eas=1;AF_eas=0.0005\n",
    # multiallelic (same CHROM,POS twice; both dropped)
    "chr1\t300\trs2\tG\tA\t.\tPASS\tAC=1;AF=0.001;AC_eas=0;AF_eas=0\n",
    "chr1\t300\trs3\tG\tT\t.\tPASS\tAC=2;AF=0.002;AC_eas=0;AF_eas=0\n",
    # biallelic SNP missing AC_eas/AF_eas (skipped when --pop eas)
    "chr1\t400\trs4\tA\tG\t.\tPASS\tAC=5;AF=0.005\n",
    # biallelic SNP (kept)
    "chr1\t500\trs5\tC\tT\t.\tPASS\tAC=7;AF=0.007;AC_eas=2;AF_eas=0.0009\n",
]


def _write_toy(tmp: Path) -> Path:
    p = tmp / "toy.vcf.gz"
    with gzip.open(p, "wt") as o:
        o.writelines(TOY_VCF)
    return p


class PickTests(unittest.TestCase):
    def test_present(self):
        self.assertEqual(pick("AC=3;AF=0.5;AC_eas=1", "AF"), "0.5")

    def test_population_suffix(self):
        self.assertEqual(pick("AC=3;AF=0.5;AC_eas=1;AF_eas=0.1", "AF_eas"), "0.1")

    def test_missing(self):
        self.assertIsNone(pick("AC=3;AF=0.5", "AF_eas"))

    def test_does_not_substring_match(self):
        # AF_eas must not match when only AF is present
        self.assertIsNone(pick("AC=3;AF=0.5", "AF_eas"))


class ParsePopafEasTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.input = _write_toy(self.tmp)
        self.output = self.tmp / "out.eas.vcf.gz"

    def tearDown(self):
        self._tmp.cleanup()

    def test_counts(self):
        result = parse_popaf_vcf(
            input_path=str(self.input),
            output_path=str(self.output),
            pop="eas",
            workdir=str(self.tmp / "wd"),
        )
        self.assertEqual(result["kept"], 2)
        self.assertEqual(result["skipped"], 1)  # rs4 missing AF_eas
        self.assertEqual(result["indels"], 1)
        self.assertEqual(result["multi"], 2)    # rs2 + rs3
        self.assertEqual(result["verified"], 2)
        self.assertEqual(result["ac_key"], "AC_eas")
        self.assertEqual(result["af_key"], "AF_eas")

    def test_output_obeys_gprism_contract(self):
        """Output must round-trip through gprism.io.parser.load_snp_database."""
        parse_popaf_vcf(
            input_path=str(self.input),
            output_path=str(self.output),
            pop="eas",
            workdir=str(self.tmp / "wd"),
        )
        records = []
        with gzip.open(self.output, "rt") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                # This is the exact expression used by
                # gprism.io.parser.load_snp_database
                popaf = float(fields[7].split(";")[1].split("=")[1])
                records.append((fields[0], fields[1], fields[3], fields[4], popaf))
        self.assertEqual(records, [
            ("chr1", "200", "T", "C", 0.0005),
            ("chr1", "500", "C", "T", 0.0009),
        ])

    def test_headers_preserved(self):
        parse_popaf_vcf(
            input_path=str(self.input),
            output_path=str(self.output),
            pop="eas",
            workdir=str(self.tmp / "wd"),
        )
        headers = []
        with gzip.open(self.output, "rt") as fh:
            for line in fh:
                if line.startswith("#"):
                    headers.append(line)
        self.assertEqual(headers, [
            "##fileformat=VCFv4.2\n",
            "##contig=<ID=chr1,length=100>\n",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n",
        ])

    def test_workdir_cleaned(self):
        wd = self.tmp / "wd"
        parse_popaf_vcf(
            input_path=str(self.input),
            output_path=str(self.output),
            pop="eas",
            workdir=str(wd),
        )
        self.assertFalse(wd.exists())

    def test_keep_workdir(self):
        wd = self.tmp / "wd_keep"
        parse_popaf_vcf(
            input_path=str(self.input),
            output_path=str(self.output),
            pop="eas",
            workdir=str(wd),
            cleanup=False,
        )
        self.assertTrue(wd.exists())
        self.assertTrue((wd / "header.vcf").exists())
        self.assertTrue((wd / "biallelic.vcf").exists())


class ParsePopafUniversalTests(unittest.TestCase):
    def test_universal_ac_af(self):
        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            inp = _write_toy(tmp)
            out = tmp / "out.universal.vcf.gz"
            result = parse_popaf_vcf(
                input_path=str(inp),
                output_path=str(out),
                pop="",
                workdir=str(tmp / "wd"),
            )
            self.assertEqual(result["ac_key"], "AC")
            self.assertEqual(result["af_key"], "AF")
            # All 3 biallelic SNPs kept: rs1, rs4, rs5
            self.assertEqual(result["kept"], 3)
            self.assertEqual(result["skipped"], 0)


class ParsePopafTypoTests(unittest.TestCase):
    def test_nonexistent_population_skips_all(self):
        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            inp = _write_toy(tmp)
            out = tmp / "out.xyz.vcf.gz"
            result = parse_popaf_vcf(
                input_path=str(inp),
                output_path=str(out),
                pop="xyz",
                workdir=str(tmp / "wd"),
            )
            self.assertEqual(result["kept"], 0)
            self.assertEqual(result["verified"], 0)
            # All 3 biallelic SNPs skipped because AC_xyz/AF_xyz don't exist
            self.assertEqual(result["skipped"], 3)


class ParsePopafRoundTripTests(unittest.TestCase):
    """Verify output can actually be loaded by gprism.io.parser.load_snp_database."""

    def test_load_snp_database(self):
        from gprism.io.parser import load_snp_database

        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            inp = _write_toy(tmp)
            out = tmp / "out.eas.vcf.gz"
            parse_popaf_vcf(
                input_path=str(inp),
                output_path=str(out),
                pop="eas",
                workdir=str(tmp / "wd"),
            )
            # The toy VCF has PopAFs well below 0.1, so pass an explicit wide
            # range to disable filtering (popaf_range=None now applies the
            # default (0.1, 0.9) for any db_type — unified policy).
            db = load_snp_database(str(out), db_type="korea4k",
                                   popaf_range=(0.0, 1.0))
        self.assertEqual(db, {
            ("chr1", "200"): ("T", "C", 0.0005),
            ("chr1", "500"): ("C", "T", 0.0009),
        })


if __name__ == "__main__":
    unittest.main()
