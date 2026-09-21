"""Check the alternative-circuit certificates, batch coverage and quotient series."""
import json
from verify_results import verify_integrity
from alternatives.verification import verify_d3, verify_rp2_sixth, verify_fold_sixth, verify_chan_v1, verify_chan_four_round
from alternatives.verification import verify_comparisons, verify_chan_zx_records


def main():
    report = {"integrity": verify_integrity()}
    for name in ("rp2-d3", "fold-d3", "chan-v1-d3"):
        report[name] = verify_d3(name)
    report["rp2-d5"] = verify_rp2_sixth()
    report["fold-d5"] = verify_fold_sixth()
    report["chan-v1-d5"] = verify_chan_v1()
    report["chan-four-round-d5"] = verify_chan_four_round()
    report["comparisons"] = verify_comparisons()
    report["zx"] = verify_chan_zx_records()
    report["scope"] = "Saved records, arithmetic and coverage; native contractions are not rerun."
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
