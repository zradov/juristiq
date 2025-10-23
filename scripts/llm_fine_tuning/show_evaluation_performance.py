import logging
from argparse import ArgumentParser
from juristiq.config.logging_config import configure_logging
from juristiq.inference.performance import get_evaluation_performance


configure_logging()
logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(description="Showing the overall performance of selected evaluation jobs.")

    parser.add_argument("-n", "--job-name-filter",
                         required=True,
                         help="A string value that the evaluation jobs need to have in the names in order to be selected.")
    
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    logger.info(f"Getting evaluation performance score for evaluation jobs '{args.job_name_filter}' ...")
    score = get_evaluation_performance(args.job_name_filter)
    logger.info(f"Overall score for '{args.job_name_filter}' evaluation jobs is: {score:.2f}")

