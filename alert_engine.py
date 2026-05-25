import argparse

from application.services.analysis_service import run_once


def build_parser():
    parser = argparse.ArgumentParser(description="Telegram alert engine entrypoint")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--period", default="15m")
    parser.add_argument("--notify-telegram", action="store_true")
    parser.add_argument("--verify-output", default="")
    parser.add_argument("--verify-include-results", action="store_true", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run_once(
        args.symbols,
        str(args.period or "15m"),
        bool(args.notify_telegram),
        verify_output=str(args.verify_output or ""),
        verify_include_results=args.verify_include_results,
    )


if __name__ == "__main__":
    raise SystemExit(main())
