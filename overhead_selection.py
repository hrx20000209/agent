import argparse

from overhead_benchmark_common import add_common_args, add_phash_args, run_overhead_case


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_phash_args(parser)
    args = parser.parse_args()
    run_overhead_case(args, case_name="selection")


if __name__ == "__main__":
    main()
