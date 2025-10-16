import os
import json
import logging
import argparse
from multiprocessing import Pool
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.utils.math import cosine_similarity


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(description="CUAD Utilities")
    parser.add_argument("-a", "--annots_file", type=str, required=True, help="Path to the JSON file containing CUAD annotations")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Path to the directory where all JSON chunks will be stored")
    args = parser.parse_args()

    return args


def chunk_cuad(annots_file: str, 
               output_dir: str) -> None:
    
    with open(annots_file, "r") as fp:
        json_data = json.load(fp)
        data_items = json_data["data"]
    
        for data_item in data_items:
            logger.info(f"Processing {data_item['title']} ...")
            paragraphs = []

            for paragraph in data_item["paragraphs"]:
                paragraphs.append(paragraph)

            output_file = os.path.join(output_dir, f"{data_item['title']}.json")

            save_to_json({"paragraphs": paragraphs}, output_file)


def save_to_json(data: dict, output_file: str):

    with open(output_file, "w") as fp:
        json.dump(data, fp)
    
    logger.info(f"Data saved to '{output_file}'.")
