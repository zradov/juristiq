import argparse
from cuad_dataset_balancer import CuadDatasetBalancer


def parse_args() -> argparse.Namespace:
    """
    Parses and returns CLI options.

    Returns:
        argparse.Namespace object containing the CLI options and values.
    """
    parser = argparse.ArgumentParser(description="Balanced CUAD review annotations based on the review label values.")
    parser.add_argument("-r", "--reviewed-cuad-annots-path", 
                        type=str, 
                        required=True, 
                        help="Path to a local folder or to the S3 bucket containing the reviewed CUAD annotations.")
    parser.add_argument("-o", "--balanced-cuad-annots-path", 
                        type=str, 
                        required=True, 
                        help="Path to a local folder or to the S3 bucket containing balanced annotations.")
    args = parser.parse_args()

    return args


def main(reviewed_annots_path: str, 
         balanced_annots_path: str):
    balancer = CuadDatasetBalancer(reviewed_annots_path, balanced_annots_path)
    balancer.balance()


if __name__ == "__main__":
    args = parse_args()
    main(args.reviewed_cuad_annots_path, args.balanced_cuad_annots_path)
