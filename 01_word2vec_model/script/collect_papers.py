import argparse
import logging
import pandas as pd
from matnexus import PaperCollector as spc


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def normalize_results(df):
    if df is None:
        return pd.DataFrame()

    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame(df)

    if not df.empty and "eid" not in df.columns:
        df = df.reset_index().rename(columns={"index": "eid"})

    expected_cols = ["eid", "source", "title", "year", "citedby_count", "abstract"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = "" if col != "citedby_count" else 0
    return df[expected_cols]


def parse_arxiv_entry(entry):
    atom = "{http://www.w3.org/2005/Atom}"

    def _text(tag):
        node = entry.find(f"{atom}{tag}")
        return node.text if node is not None and node.text is not None else ""

    published = _text("published")
    return {
        "eid": _text("id"),
        "source": "arXiv",
        "title": _text("title"),
        "year": published[:4] if published else "",
        "citedby_count": 0,
        "abstract": _text("summary"),
    }


def resilient_collect(config_path, query, arxiv_limit):
    rows = []
    logger = logging.getLogger(__name__)

    # Retrieve Scopus records one-by-one so a single bad EID does not abort all results.
    try:
        import pybliometrics.scopus
        from pybliometrics.scopus import AbstractRetrieval, ScopusSearch

        pybliometrics.scopus.init(config_path=config_path)
        eids = ScopusSearch(query["Scopus"]).get_eids() or []
        print(f"Resilient mode: Scopus returned {len(eids)} EIDs.")

        for eid in eids:
            try:
                abstract = AbstractRetrieval(eid, view="FULL")
                rows.append(
                    {
                        "eid": getattr(abstract, "eid", eid),
                        "source": "Scopus",
                        "title": getattr(abstract, "title", "") or "",
                        "year": (getattr(abstract, "coverDate", "") or "")[:4],
                        "citedby_count": getattr(abstract, "citedby_count", 0) or 0,
                        "abstract": getattr(abstract, "abstract", "") or "",
                    }
                )
            except Exception as exc:
                logger.warning("Skipping Scopus EID %s due to retrieval error: %s", eid, exc)
    except Exception as exc:
        logger.error("Resilient Scopus collection failed: %s", exc)

    # Cap arXiv requests to reduce API throttling for unbounded pulls.
    try:
        entries = spc.ArxivDataSource().search(query["arXiv"], limit=arxiv_limit)
        print(f"Resilient mode: arXiv returned {len(entries)} entries (limit={arxiv_limit}).")
        rows.extend(parse_arxiv_entry(entry) for entry in entries)
    except Exception as exc:
        logger.error("Resilient arXiv collection failed: %s", exc)

    return pd.DataFrame(rows)


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
        "--openaccess",
        type=parse_bool,
        required=True,
        help="Search for open access papers",
    )
    parser.add_argument(
        "--arxiv_limit",
        type=int,
        default=200,
        help="Maximum number of arXiv records to fetch in resilient fallback mode",
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
    results = collector.results
    if results.empty:
        print("Primary collection returned no rows. Retrying with resilient fallback...")
        results = resilient_collect(
            config_path=args.config_path,
            query=query,
            arxiv_limit=args.arxiv_limit,
        )

    results = normalize_results(results)
    results.to_csv(args.output_path, index=False)
    print(f"Wrote {len(results)} papers to {args.output_path}")


if __name__ == "__main__":
    main()
