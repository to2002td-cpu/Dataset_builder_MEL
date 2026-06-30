"""
Generate an HTML annotation analysis report.

Usage:
    python generate_report.py <annotations_dir> [--output report.html]

<annotations_dir> must contain one sub-folder per annotator,
each with a user_state.json file produced by the annotation tool.
"""

import argparse
from pathlib import Path

import annotation.state.html_report as html_report
import annotation.state.loader as loader
import annotation.state.stats as stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an HTML annotation analysis report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python generate_report.py output/annotations\n"
            "  python generate_report.py output/annotations --output results/report.html"
        ),
    )
    parser.add_argument(
        "annotations_dir",
        type=Path,
        help="Folder containing one sub-folder per annotator (each with user_state.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "report.html",
        metavar="FILE",
        help="Output HTML file (default: report.html next to this script).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.annotations_dir.exists():
        print(f"Error: directory not found: {args.annotations_dir}")
        raise SystemExit(1)

    annotators = loader.load_annotators(args.annotations_dir)

    if not annotators:
        print(f"Error: no user_state.json files found in {args.annotations_dir}")
        raise SystemExit(1)

    annotation_stats = stats.compute(annotators)
    html = html_report.render(annotators, annotation_stats)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")

    _print_summary(annotation_stats)
    print(f"\nReport written to: {args.output}")


def _print_summary(s: stats.AnnotationStats) -> None:
    from collections import Counter
    majority_counts    = Counter(r.majority for r in s.instance_results)
    unanimous_count    = sum(1 for r in s.instance_results if r.unanimous)
    disagreement_count = len(s.instance_results) - unanimous_count

    print(f"Annotators  : {len(s.annotator_ids)}")
    print(f"Instances   : {len(s.all_instances)}")
    print(f"Unanimous   : {unanimous_count}")
    print(f"Disagreements: {disagreement_count}")
    print(f"Majority YES: {majority_counts.get('YES', 0)}")
    print(f"Majority NO : {majority_counts.get('NO', 0)}")
    print(f"No majority : {majority_counts.get('NO MAJORITY', 0)}")
    print(f"Mean kappa  : {s.mean_kappa} ({stats.interpret_kappa(s.mean_kappa)})")


if __name__ == "__main__":
    main()
