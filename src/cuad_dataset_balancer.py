import json
import logging
from enum import Enum
from pathlib import Path
from random import sample
from llm_utils import (
    get_missing_clause_prompt_builder, 
    get_rephrase_text_prompt_builder,
    get_clause_prompt_builder
)
from genai_output_parsers import (
    TitleSubtitleParser,
    ClauseAugmentationParser
)
from data_utils import DataBatch
from annots_utils import get_hash
from genai_utils import log_user_balance
from genai_clients import GenAIClientFactory
from logging_config import configure_logging
from remote_paraphraser import RemoteParaphraser
from genai_exceptions import (
    GenAIAuthError, 
    GenAIInsufficientBalanceError
)
from typing import TypedDict, Any, Callable, Tuple, NamedTuple


configure_logging()
logger = logging.getLogger(__name__)


class ReviewLabelsExpectedPct(Enum):
    # The expected percentage range of different review labels in the balanced dataset.
    Missing = (0.2, 0.3)
    Compliant = (0.35, 0.45)
    Risky = (0.2, 0.3)
    LackOfRequiredData = (0.05, 0.1)


class CuadAnnots(TypedDict):
    # a dictionary where keys are different review labels and values
    # lists of annotations related to a specific review label.
    Missing: list[dict]
    Compliant: list[dict]
    Risky: list[dict]
    LackOfRequiredData: list[dict]


class AugmentationInfo(NamedTuple):
    # total number of annotations to generate
    total_augmentations: int
    # number of augmented annotations to create per original annotation
    augmentations_per_annotation: int
    # reduce per_annotations_count after processing this number of original annotations
    reduce_augmentations_after: int
    

class CuadDatasetBalancer:
    """
    A class to balance the compliance review annotations dataset by oversampling the annotations of the 
    underrepresented review labels using GenAI providers to rephrase the answers titles and replace the original 
    answers in the original context with the new phrases.
    The balanced dataset will have the expected percentage of different review labels as defined in the 
    ReviewLabelsExpectedPct enum.
    """
    def __init__(self, 
                 src_annots_path: str, 
                 dest_annots_path: str,
                 genai_provider_name: str="deep_seek"):
        """
        Initializes the instance of the CuadDataBalancer class.

        Args:
            src_annots_path: a local file system path to the folder with the compliance review annotations that require data balancing.
            dest_annots_path: a local file system path to the folder where the original and the augmented annotations will be stored.

        Returns:
            None
        """
        self.src_annots_path = Path(src_annots_path)
        self.dest_annots_path = Path(dest_annots_path)
        self.dest_annots_path.mkdir(parents=True, exist_ok=True)
        self.genai_provider_name = genai_provider_name
        self.paraphraser = RemoteParaphraser(genai_provider_name=genai_provider_name)
        self.missing_clause_aug_prompt_builder = get_missing_clause_prompt_builder()
        self.rephrase_text_prompt_builder = get_rephrase_text_prompt_builder()
        self.clause_prompt_builder = get_clause_prompt_builder()
        self.clause_output_parser = ClauseAugmentationParser()
        self.title_subtitle_output_parser = TitleSubtitleParser()


    def _get_review_label_key(self, review_label: str) -> str:
        """
        Converts the review label to a key format used in the CuadAnnots dictionary.

        Args:
            review_label: a review label string.

        Returns:
            a review label string converted to a key format.
        """
        return "".join([s.capitalize() for s in review_label.split()])

    
    def _load_annots_from_folder(self, src_path: str) -> CuadAnnots:
        """
        Loads and returns the annotations, grouped by the review label, from the specified folder.

        Args:
            src_path: a path to the folder with the annotations.

        Returns:
            a dictionary where keys are different review labels and values lists of annotations related to a specific review label.
        """
        cuad_annots = CuadAnnots()

        for annot_path in src_path.rglob("*.json"):
            annot = json.loads(annot_path.read_text(encoding="utf-8"))
            annot["hash"] = annot_path.name
            review_label = annot["review_label"]
            review_label_annots = cuad_annots.setdefault(self._get_review_label_key(review_label), [])
            review_label_annots.append(annot)

        cuad_annots = {k:self._remove_duplicates(v) for k, v in cuad_annots.items()}

        return cuad_annots


    def _load_annots(self) -> CuadAnnots:
        """
        Loads and returns the annotations, grouped by the review label, 
        from the source and the destination folders combined.

        Returns:
            a dictionary where keys are different review labels and values
            lists of annotations related to a specific review label.
        """
        src_cuad_annots = self._load_annots_from_folder(self.src_annots_path)
        dest_cuad_annots = self._load_annots_from_folder(self.dest_annots_path)

        return src_cuad_annots, dest_cuad_annots


    def _create_missing_augmented_annots(self,
                                         original_annot: dict, 
                                         new_phrases: dict[str, list[str]]) -> list[dict] | None:
        """
        Creates and returns a list of augmented annotations for the original annotation
        that has the "Missing" review label.
        
        Args:
            original_annot: an original annotation dictionary.
            new_phrases: a dictionary where keys are clause types and values are lists of new phrases for the specific clause type.

        Returns:
            a list of augmented annotations or None if the clause type of the original annotation
            is not found in the new_phrases dictionary. 
        """
        augmented_annots = []
        original_annot_clause = original_annot["clause_type"]

        if original_annot_clause not in new_phrases:
            logging.warning(f"The '{original_annot_clause} clause type not found in the augmented clause types {list(new_phrases.keys())}.")
            return
        for item in new_phrases[original_annot_clause]:
            new_annot = original_annot.copy()
            # The try/except block added as a temporary work around for the issue 
            # https://github.com/zradov/juristiq/issues/1.
            try:
                new_annot["suggested_redline"] = item["Suggested Redline"]
                new_annot["rationale"] = item["Rationale"]
            except KeyError as ex:
                logging.error(f"Key {ex.args[0]} not found in {item}.")
                continue
            augmented_annots.append(new_annot)

        return augmented_annots


    def _create_augmented_annots(self,
                                 original_annot: dict, 
                                 new_phrases: dict[str, list[str]]) -> list[dict]:
        """
        Creates and returns a list of augmented annotations for the original annotation
        that has the "Compliant", "Risky" or "Lack of Required Data" review label.

        Args:
            original_annot: an original annotation dictionary.
            new_phrases: a dictionary where keys are original answers and values are lists of new phrases for the specific answer.

        Returns:
            a list of augmented annotations.
        """
        versions_count = len(list(new_phrases.values())[0])
        augmented_annots = []
        answers_boundaries = {}
        answers_start = []
        answers = sorted(original_annot["answers"], key=lambda i: len(i), reverse=True)

        for answer in answers:
            start_idx = original_annot["context"].find(answer)
            if start_idx in answers_start:
                logging.warning(f"Index {start_idx} for answer '{answer}' in the context '{original_annot["context"]}' already exists.")
                continue
            answers_start.append(start_idx)
            answers_boundaries[answer] = (start_idx, start_idx + len(answer))
        
        for version_idx in range(versions_count):
            new_annot = original_annot.copy()
            new_context = []
            start_pos = 0

            for answer, (start_idx, end_idx) in answers_boundaries.items():
                # Temporary workaround for the issue https://github.com/zradov/juristiq/issues/1
                if version_idx >= len(new_phrases.get(answer, [])):
                    logger.warning(f"The version index {version_idx} is out of range for the answer '{answer}'.")
                    return augmented_annots
                new_context.append(original_annot["context"][start_pos:start_idx])
                if answer not in new_phrases:
                    logger.warning(f"The answer '{answer}' not found the new phrases list.")
                    continue
                new_phrase = new_phrases[answer][version_idx]
                new_context.append(new_phrase)
                start_pos = end_idx

            new_context.append(original_annot["context"][start_pos:])
            new_annot["context"] = "".join(new_context)
            new_annot["originated_from"] = original_annot["hash"]
            augmented_annots.append(new_annot)

        return augmented_annots
    

    def _get_prompt_builder(self, review_label) -> Callable:
        """
        Returns a prompt builder function based on the review label.

        Args:
            review_label: a review label string.
        
        Returns:
            a prompt builder function.
        """
        return self.missing_clause_aug_prompt_builder \
            if review_label == "Missing" \
            else self.rephrase_text_prompt_builder if review_label == "Compliant" \
            else self.clause_prompt_builder


    def _get_output_parser(self, review_label: str) -> Tuple[Callable, int]:
        """
        Returns an output parser function and item size based on the review label.

        Args:
            review_label: a review label string.

        Returns:
            a tuple containing an output parser function and item size.
        """
        return (self.clause_output_parser, 2) \
            if review_label == "Missing" \
            else (self.title_subtitle_output_parser, None) \
            if review_label == "Compliant" \
            else (self.clause_output_parser, 4)
    

    def _process_batch(self, 
                       annots: list[dict], 
                       batch: DataBatch,
                       versions_count: int) -> list[dict]:
        """
        Processes a batch of annotations by generating new phrases using the paraphraser
        and creating augmented annotations.

        Args:
            annots: a list of original annotations corresponding to the data in the batch.
            batch: a DataBatch object containing the data for augmentation.
            versions_count: the number of augmented versions to create for each original annotation.

        Returns:
            a list of original and augmented annotations.
        """
        augmented_annots = []

        prompt_builder = self._get_prompt_builder(annots[0]["review_label"])
        output_parser, item_size = self._get_output_parser(annots[0]["review_label"])
        new_phrases = self.paraphraser.rephrase(prompt_builder,
                                                output_parser,
                                                versions_count,
                                                item_size=item_size,
                                                data_batch=batch)
        for temp_annot in annots:
            augmented_annots.append(temp_annot)
            if temp_annot["review_label"] == "Missing":
                temp_aug_annots = self._create_missing_augmented_annots(temp_annot, new_phrases)
            else:
                temp_aug_annots = self._create_augmented_annots(temp_annot, new_phrases)
            if temp_aug_annots:
                augmented_annots.extend(temp_aug_annots)
                self._save_annots([temp_annot] + temp_aug_annots)

        return augmented_annots


    def _get_data_for_augmentation(self, annot: dict) -> Any:
        """
        Prepares and returns data from the annotation required for augmentation.

        Args:
            annot: an annotation dictionary.

        Returns:
            data required for augmentation.
        """
        def get_data(annot, answer=""):
            return {
                "answer": answer, 
                "clause_type": annot["clause_type"],
                "review_label": annot["review_label"],
                "policy_text": annot["policy_text"],
                "context": annot["context"],
                "rationale": annot["rationale"],
                "suggested_redline": annot["suggested_redline"]
            }
        if annot["review_label"] == "Missing":
            return {
                k: annot[k] for k in ["clause_type", "policy_text"]
            }
        if annot["review_label"] == "Compliant":
            return [a for a in annot["answers"]]
        if annot["answers"]:
            return [
                get_data(annot, a)
                for a in annot["answers"]
            ]
        return get_data(annot)


    def _oversample(self, 
                    annots: list[dict], 
                    target: int,
                    augmentation_info: AugmentationInfo,
                    batch_size=10) -> list[dict]:
        """
        Oversamples the provided annotations so that the count of the annotations after oversampling 
        is equal to the target count. Oversampling is done by rephrasing the answers titles, replacing
        the original answers in the original context with the new phrases and creating new annotation
        sample. The new annotation samples will contain references to the original sample and 
        semantic similarity score between the original and the augmented annotation sample.

        Args:
            annots: a list of annotations having the same review label
            target: the expected count of annotations after oversampling
            batch_size: recommended number of titles to put into a single prompt when doing rephrasing

        Returns:
            an oversampled annotations list
        """
        augmented_annots = []
        batch = DataBatch(batch_size=batch_size,
                          data_store_type=list if annots[0]["review_label"] == "Missing" else set)
        temp_annots = []

        try:
            for i, annot in enumerate(annots):
                # The code is temporary commented because of the issue https://github.com/zradov/juristiq/issues/1
                #if self._is_augmented_annot_exist(annot):
                #    logger.info(f"The annotation {self._get_annot_hash(annot)}.json already exists.")
                #    continue
                num_augments = augmentation_info.augmentations_per_annotation \
                    if i < augmentation_info.reduce_augmentations_after \
                    else augmentation_info.augmentations_per_annotation-1
                batch.add(self._get_data_for_augmentation(annot))
                temp_annots.append(annot)
                if batch.is_full():
                    temp_annots = self._process_batch(temp_annots, batch, num_augments)
                    augmented_annots.extend(temp_annots)
                    temp_annots.clear()
                    batch.clear()

            if batch.has_items():
                temp_annots = self._process_batch(temp_annots, batch, num_augments)
                augmented_annots.extend(temp_annots)

            return augmented_annots
        except GenAIAuthError as err:
            logger.error(f"Authentication failed. {err}.")
            raise
        except GenAIInsufficientBalanceError as err:
            logger.error(f"Insufficient balance in the user account for making the request. {err}.")
            log_user_balance(self.genai_provider_name)
            raise
        except Exception as err:
            logger.error(f"Failed to oversample the annotations. {err}.")
            raise


    def _get_annot_hash(self, annot: dict) -> str:
        """
        Returns a hash string for the specified annotation.

        Args:
            annot: an annotation dictionary.

        Returns:
            a hash string for the specified annotation.
        """
        hash_data = {k: annot[k] for k in ["question", "context", "policy_id", "clause_type", "suggested_redline", "rationale"] }
        annot_hash = get_hash([hash_data])
        
        return annot_hash


    def _get_annot_save_path(self, annot: dict) -> Path:
        """
        Returns a path where the specified annotation should be saved.

        Args:
            annot: an annotation dictionary.

        Returns:
            a path where the specified annotation should be saved.
        """
        annot_hash = self._get_annot_hash(annot)
        path = Path(self.dest_annots_path) / f"{annot_hash}.json"
        
        return path


    def _is_augmented_annot_exist(self, annot: dict) -> bool:
        """
        Checks if the specified annotation already exists in the destination folder.

        Args:
            annot: an annotation dictionary.

        Returns:
            True if the specified annotation already exists in the destination folder, False otherwise.
        """
        dest_path = self._get_annot_save_path(annot)
        
        return dest_path.exists()
    

    def _save_annots(self, annots: list[dict]) -> None:
        """
        Saves the specified annotations in the destination folder.

        Args:
            annots: a list of annotations.
        """
        for annot in annots:
            dest_path = self._get_annot_save_path(annot)
            if dest_path.exists():
                continue
            dest_path.write_text(json.dumps(annot), encoding="utf-8")

            
    def _remove_duplicates(self, annots: list[dict]) -> list[dict]:
        """
        Removes duplicate annotations from the specified list of annotations.

        Args:
            annots: a list of annotations.

        Returns:
            a list of unique annotations.
        """
        unique_annots = []
        unique_hashes = set([])

        for annot in annots:
            annot_hash = self._get_annot_hash(annot)
            if annot_hash not in unique_hashes:
                unique_annots.append(annot)
                unique_hashes.add(annot_hash)

        if len(unique_annots) != len(annots):
            logger.info(f"Removed {len(annots)-len(unique_annots)} duplicate annotations.")    

        return unique_annots


    def _get_expected_annots_counts(self,
                                    annots: CuadAnnots, 
                                    annots_target_count: int) -> dict[str, AugmentationInfo]:
        
        """
        Calculates and returns the expected counts of annotations for different review labels
        based on the expected percentage ranges defined in the ReviewLabelsExpectedPct enum.

        Args:
            annots: a dictionary where keys are different review labels and values lists of annotations related to a specific review label.
            annots_target_count: the expected total count of annotations after balancing.

        Returns:
            a dictionary where keys are different review labels and values are expected counts of annotations related to a specific review label.
        """
        result = {}
        total_annots = sum([len(annots) for _, annots in annots.items()])
        remaining = annots_target_count

        for review_label, review_label_annots in annots.items():
            low_pct, max_pct = ReviewLabelsExpectedPct[review_label].value
            annots_pct = len(review_label_annots) / total_annots
            expected_annots_pct = max(min(annots_pct, max_pct), low_pct)
            expected_annots_count = round(total_annots * expected_annots_pct)
            result[review_label] = expected_annots_count
            remaining -= expected_annots_count
        
        while remaining != 0:
            if abs(remaining) == 1:
                # If there is only one remaining extra annotation increase the number
                # of expected annotations for first review label.
                result[list(result.keys())[0]] += remaining
                break

            temp_remaining = remaining
            for review_label in result:
                low_pct, _ = ReviewLabelsExpectedPct[review_label].value
                extra_annots_count = round(low_pct * remaining) 
                result[review_label] += extra_annots_count
                temp_remaining -= extra_annots_count

            remaining = temp_remaining

        result = {
            k: AugmentationInfo(
                total_augmentations=v,
                augmentations_per_annotation=int(annots_target_count / len(annots)) + 1,
                reduce_augmentations_after=annots_target_count % len(annots))
            for k, v in result.items()
        }        

        return result


    def _log_current_annots_count(self, annots: CuadAnnots) -> None:
        """
        Logs the current count of annotations per review label.

        Args:
            annots: a dictionary where keys are different review labels and values lists of annotations related to a specific review label.
        """
        print()
        print("Annotations count per clause type:")
        for clause_type in annots:
            print(f"{clause_type}: {len(annots[clause_type])}")
        print()


    def balance(self, annot_target_count: int=20000) -> CuadAnnots:
        """
        It balances the annotations' dataset by oversampling or downsampling the annotations of the minority 
        or majority review labels using GenAI providers to rephrase the answers titles and replace the original 
        answers in the original context with the new phrases. The balanced dataset will have the expected percentage 
        of different review labels as defined in the ReviewLabelsExpectedPct enum.
        
        Args:
            annot_target_count: the expected total count of annotations after balancing (default: 20000).
        
        Returns:
            a dictionary where keys are different review labels and values lists of balanced annotations related 
            to a specific review label.
        """
        balanced_annots = {}
        src_annots, dest_annots = self._load_annots()
        all_annots = {k: v + dest_annots.get(k, []) 
                      for k, v in src_annots.items()}
        self._log_current_annots_count(all_annots)
        augmentation_infos = self._get_expected_annots_counts(dest_annots, annot_target_count)

        for review_label in src_annots:
            annots = dest_annots[review_label] \
                if review_label in dest_annots \
                else src_annots[review_label]
            if len(annots) > augmentation_infos[review_label].total_augmentations:
                temp_annots = sample(annots, augmentation_infos[review_label].total_augmentations)
                balanced_annots[review_label] = temp_annots
                self._save_annots(temp_annots)
            elif len(annots) < augmentation_infos[review_label].total_augmentations:
                self._save_annots(annots)
                annots_to_generate = augmentation_infos[review_label].total_augmentations - len(annots)
                generated_annots_count = 0
                # the code is put inside the while loop because of the issue 
                # https://github.com/zradov/juristiq/issues/1
                while generated_annots_count < annots_to_generate:
                    if review_label not in balanced_annots:
                        balanced_annots[review_label] = []
                    # When doing oversampling take the original annotations from the source folder.
                    temp_annots = self._oversample(src_annots[review_label], 
                                                   annots_to_generate,
                                                   augmentation_infos[review_label])
                    balanced_annots[review_label].extend(temp_annots)
                    generated_annots_count += len(temp_annots)
                    annots_to_generate -= generated_annots_count
                        
            else:
                balanced_annots[review_label] = annots

        return balanced_annots


if __name__ == "__main__":
    c = CuadDatasetBalancer(src_annots_path=r"C:\projects\ML\juristiq\output\juristiq-cuad-reviewed",
                            dest_annots_path=r"C:\projects\ML\juristiq\output\juristiq-cuad-balanced")
    
    """
    annots = c._load_annots()
    expected_annots_counts = c._get_expected_annots_counts(annots, 4000)
    print(expected_annots_counts)
    """
    c.balance()
