import argparse
from matnexus import PaperCollector as spc


def parse_args():
    parser = argparse.ArgumentParser(description="Collect papers from multiple sources")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the configuration file for ScopusDataSource",
    )
    parser.add_argument(
        "--keywords", type=str, required=True, help="Keywords for the paper query"
    )
    parser.add_argument(
        "--startyear", type=int, help="Start year for the paper query (optional)"
    )
    parser.add_argument(
        "--endyear", type=int, required=True, help="End year for the paper query"
    )
    parser.add_argument(
        "--openaccess", type=bool, required=True, help="Search for open access papers"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for the collected papers CSV file",
    )

    return parser.parse_args()


def main():
    print("Collecting papers from multiple sources...")
    args = parse_args()

    ScopusDataSource = spc.ScopusDataSource(config_path=args.config_path)
    ArxivDataSource = spc.ArxivDataSource()

    sources = [ScopusDataSource, ArxivDataSource]

    # Build query, only include startyear if provided
    query = spc.MultiSourcePaperCollector.build_query(
        keywords=args.keywords,
        startyear=args.startyear if args.startyear else None,
        endyear=args.endyear,
        openaccess=args.openaccess,
    )

    collector = spc.MultiSourcePaperCollector(sources, query)
    collector.collect_papers()
    collector.results.to_csv(args.output_path)


if __name__ == "__main__":
    main()
