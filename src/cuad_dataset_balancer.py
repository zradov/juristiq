import json
import logging
from enum import Enum
from pathlib import Path
from random import sample, shuffle
from llm_utils import (
    get_missing_clause_prompt_builder, 
    get_rephrase_text_prompt_builder,
    get_lack_of_required_data_clause_prompt_builder,
    get_risky_clause_prompt_builder
)
from genai_output_parsers import (
    Parser,
    TitleSubtitleParser,
    ClauseAugmentationParser
)
from data_utils import DataBatch
from annots_utils import get_hash
from genai_utils import log_user_balance
from logging_config import configure_logging
from remote_paraphraser import RemoteParaphraser
from genai_exceptions import (
    GenAIAuthError, 
    GenAIInsufficientBalanceError
)
from genai_exceptions import GenAITokenLimitError   
from typing import cast, TypedDict, Any, Callable, Tuple, NamedTuple


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

    @classmethod
    def default(cls) -> "CuadAnnots":
        return {
            "Missing": [],
            "Compliant": [],
            "Risky": [],
            "LackOfRequiredData": []
        }


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
        self._src_annots_path = Path(src_annots_path)
        self._dest_annots_path = Path(dest_annots_path)
        self._dest_annots_path.mkdir(parents=True, exist_ok=True)
        self._genai_provider_name = genai_provider_name
        self._paraphraser = RemoteParaphraser(genai_provider_name=genai_provider_name)
        self._missing_clause_aug_prompt_builder = get_missing_clause_prompt_builder()
        self._rephrase_text_prompt_builder = get_rephrase_text_prompt_builder()
        self._lack_of_required_data_clause_prompt_builder = get_lack_of_required_data_clause_prompt_builder()
        self._risky_clause_prompt_builder = get_risky_clause_prompt_builder()
        self._clause_output_parser = ClauseAugmentationParser()
        self._title_subtitle_output_parser = TitleSubtitleParser()
        self._augmentation_func = {
            "Missing": self._create_missing_augmented_annots,
            "Compliant": self._create_compliant_augmented_annots,
            "Risky": self._create_risky_augmented_annots,
            "Lack of required data": self._create_lack_of_required_data_augmented_annots
        }
        

    def _get_review_label_key(self, review_label: str) -> str:
        """
        Converts the review label to a key format used in the CuadAnnots dictionary.

        Args:
            review_label: a review label string.

        Returns:
            a review label string converted to a key format.
        """
        return "".join([s.capitalize() for s in review_label.split()])

    
    def _validate_annots(self, annots: CuadAnnots) -> CuadAnnots:
        """
        Checks whether there are errors in the annotations data.

        Args:
            annots: the input annotations

        Returns:
            a valid annotations 
        """
        validated_annots = cast(dict, CuadAnnots.default())

        for review_label, review_label_annots in annots.items():
            validated_annots[review_label] = []
            if review_label == "Compliant":
                for annot in review_label_annots:
                    are_answers_found = True
                    for answer in annot["answers"]:
                        if answer not in annot["context"]:
                            are_answers_found = False
                            break
                    if are_answers_found:
                        validated_annots[review_label].append(annot)
            elif review_label == "Risky":
                for annot in review_label_annots:
                    if annot["context"].strip():
                        validated_annots[review_label].append(annot)
            else:
                validated_annots[review_label].extend(review_label_annots)

        logger.info("Count of removed invalid annots per review label:")
        for review_label, review_label_annots in annots.items():
            logger.info(f"\t{review_label}: {len(review_label_annots)-len(validated_annots[review_label])}")

        return validated_annots


    def _load_annots_from_folder(self, 
                                 src_path: Path, 
                                 shuffle: bool=False) -> CuadAnnots:
        """
        Loads and returns the annotations, grouped by the review label, from the specified folder.

        Args:
            src_path: a path to the folder with the annotations.
            shuffle: should annotations be randomly shuffled or not.
        Returns:
            a dictionary where keys are different review labels and values lists of annotations related to a specific review label.
        """
        cuad_annots = CuadAnnots.default()

        for annot_path in src_path.rglob("*.json"):
            annot = json.loads(annot_path.read_text(encoding="utf-8"))
            annot["hash"] = annot_path.name
            review_label = annot["review_label"]
            review_label_annots = cuad_annots.setdefault(self._get_review_label_key(review_label), [])
            review_label_annots.append(annot)

        cuad_annots = {k:self._remove_duplicates(v) for k, v in cuad_annots.items()}
        cuad_annots = self._validate_annots(CuadAnnots(cuad_annots))
        if shuffle:
            for review_label in cuad_annots:
                shuffle(cuad_annots[review_label])

        return cuad_annots


    def _load_annots(self) -> Tuple[CuadAnnots, CuadAnnots]:
        """
        Loads and returns the annotations, grouped by the review label, 
        from the source and the destination folders combined.

        Returns:
            a dictionary where keys are different review labels and values
            lists of annotations related to a specific review label.
        """
        src_cuad_annots = self._load_annots_from_folder(self._src_annots_path, shuffle=True)
        dest_cuad_annots = self._load_annots_from_folder(self._dest_annots_path)

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
            logger.warning(f"The '{original_annot_clause} clause type not found in the augmented clause types {list(new_phrases.keys())}.")
            return
        for item in new_phrases[original_annot_clause]:
            new_annot = original_annot.copy()
            # The try/except block added as a temporary work around for the issue 
            # https://github.com/zradov/juristiq/issues/1.
            try:
                new_annot["suggested_redline"] = item["Suggested Redline"]
                new_annot["rationale"] = item["Rationale"]
                new_annot["originated_from"] = original_annot["hash"]
            except KeyError as ex:
                logger.error(f"Key {ex.args[0]} not found in {item}.")
                continue
            augmented_annots.append(new_annot)

        return augmented_annots


    def _create_compliant_augmented_annots(self,
                                           original_annot: dict, 
                                           new_phrases: dict[str, list[str]]) -> list[dict]:
        """
        Creates and returns a list of augmented annotations for the original annotation
        that has the "Compliant" review label.

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
            #start_idx = original_annot["context"].find(answer)
            start_idx = original_annot["context"].find(answer)
            if start_idx in answers_start:
                logger.warning(f"Index {start_idx} for answer '{answer}' in the context '{original_annot["context"]}' already exists.")
                continue
            elif start_idx == -1:
                logger.warning(f"Answer {answer} not found in the context '{original_annot["context"]}'.")
                return []               
            answers_start.append(start_idx)
            answers_boundaries[answer] = (start_idx, start_idx + len(answer))

        answers_boundaries = {
            k:answers_boundaries[k] 
            for k in sorted(answers_boundaries, key=lambda k: answers_boundaries[k][0])
        }
        
        for version_idx in range(versions_count):
            new_annot = original_annot.copy()
            new_context = []
            start_pos = 0
            new_answers = []

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
                new_answers.append(new_phrase)
                start_pos = end_idx
            
            new_context.append(original_annot["context"][start_pos:])
            new_annot["answers"] = new_answers
            new_annot["context"] = "".join(new_context)
            new_annot["originated_from"] = original_annot["hash"]
            augmented_annots.append(new_annot)

        return augmented_annots
    

    def _create_risky_augmented_annots(self,
                                       original_annot: dict, 
                                       new_phrases: dict[str, list[str]]) -> list[dict] | None:
        """
        Creates and returns a list of augmented annotations for the original annotation that has the "Risky" review label.
        
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
            logger.warning(f"The '{original_annot_clause} clause type not found in the augmented clause types {list(new_phrases.keys())}.")
            return
        for item in new_phrases[original_annot_clause]:
            new_annot = original_annot.copy()
            # The try/except block added as a temporary work around for the issue 
            # https://github.com/zradov/juristiq/issues/1.
            try:
                new_annot["context"] = item["context"]
                new_annot["answers"] = item["answer"] if isinstance(item["answer"], list) else [item["answer"]]
                new_annot["suggested_redline"] = item["suggested_redline"]
                new_annot["rationale"] = item["rationale"]
                new_annot["originated_from"] = original_annot["hash"]
            except KeyError as ex:
                logger.error(f"Key {ex.args[0]} not found in {item}.")
                continue
            augmented_annots.append(new_annot)

        return augmented_annots


    def _create_lack_of_required_data_augmented_annots(self,
                                                       original_annot: dict, 
                                                       new_phrases: dict[str, list[str]]) -> list[dict] | None:
        
        return self._create_risky_augmented_annots(original_annot, new_phrases)


    def _get_prompt_builder(self, review_label) -> Callable:
        """
        Returns a prompt builder function based on the review label.

        Args:
            review_label: a review label string.
        
        Returns:
            a prompt builder function.
        """
        review_label_normalized = self._get_review_label_key(review_label)
        if review_label_normalized not in ReviewLabelsExpectedPct.__members__:
            raise ValueError(f"Unsupported review label '{review_label_normalized}'. Supported review labels: {list(ReviewLabelsExpectedPct.__members__.keys())}.")

        return self._missing_clause_aug_prompt_builder \
            if review_label_normalized == "Missing" \
            else self._rephrase_text_prompt_builder if review_label_normalized == "Compliant" \
            else self._risky_clause_prompt_builder if review_label_normalized == "Risky" \
            else self._lack_of_required_data_clause_prompt_builder
            

    def _get_output_parser(self, review_label: str) -> Tuple[Parser, int]:
        """
        Returns an output parser function and item size based on the review label.

        Args:
            review_label: a review label string.

        Returns:
            a tuple containing an output parser function and item size.
        """
        return (self._clause_output_parser, 2) \
            if review_label == "Missing" \
            else (self._title_subtitle_output_parser, None) \
            if review_label == "Compliant" \
            else (self._clause_output_parser, 4)
    

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
        batches = [batch]
        new_phrases = {}
        while batches:
            try:
                current_batch = batches.pop()
                temp_phrases = self._paraphraser.rephrase(prompt_builder,
                                                          output_parser,
                                                          versions_count,
                                                          item_size=item_size,
                                                          data_batch=current_batch)
                if temp_phrases:
                    for k, v in temp_phrases.items():
                        new_phrases.setdefault(k, []).extend(v)
            except GenAITokenLimitError as err:
                logger.error(f"Token limit exceeded. {err}.")
                if len(current_batch) == 1:
                    raise "Batch cannot have less than 1 item."
                # Split the batch in half and try again.
                half_batch_idx = len(current_batch)//2
                batches.append(current_batch[:half_batch_idx])
                batches.append(current_batch[half_batch_idx:])
            except Exception as err:
                logger.error(f"Failed to rephrase the text. {err}.")
                raise
        for temp_annot in annots:
            augmented_annots.append(temp_annot)
            temp_aug_annots = self._augmentation_func[temp_annot["review_label"]](temp_annot, new_phrases)
            if temp_aug_annots:
                augmented_annots.extend(temp_aug_annots)
                self._save_annots([temp_annot] + temp_aug_annots)

        return augmented_annots


    def _escape_special_chars(self, text: list[str]) -> list[str]:
        """
        Formats the provided list of strings by escaping special characters.

        Args:
            text: a list of strings.

        Returns:
            a list of formatted answer strings.
        """
        return [a.replace("\n", "\\n").replace("{", "{{").replace("}", "}}") 
                for a in text]


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
                "context": self._escape_special_chars([annot["context"]])[0],
                "rationale": annot["rationale"],
                "suggested_redline": annot["suggested_redline"]
            }
        if annot["review_label"] == "Missing":
            return {
                k: annot[k] for k in ["clause_type", "policy_text"]
            }
        if annot["review_label"] == "Compliant":
            return self._escape_special_chars(annot["answers"])
        if annot["answers"]:
            return [
                get_data(annot, a)
                for a in self._escape_special_chars(annot["answers"])
            ]
        return get_data(annot)


    def _oversample(self, 
                    annots: list[dict], 
                    target: int,
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
                          data_store_type=set if annots[0]["review_label"] == "Compliant" else list)
        temp_annots = []

        try:
            for i, annot in enumerate(annots):
                # The code is temporary commented because of the issue https://github.com/zradov/juristiq/issues/1
                #if self._is_augmented_annot_exist(annot):
                #    logger.info(f"The annotation {self._get_annot_hash(annot)}.json already exists.")
                #    continue
                augmentations_per_annotation=int(target / len(annots)) + 1
                reduce_augmentations_after=target % len(annots)
                num_augments = augmentations_per_annotation \
                    if i < reduce_augmentations_after \
                    else augmentations_per_annotation-1
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
            log_user_balance(self._genai_provider_name)
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
        path = Path(self._dest_annots_path) / f"{annot_hash}.json"
        
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
        total_annots = sum([len(a) for _, a in annots.items()])
        remaining = annots_target_count

        for review_label, review_label_annots in annots.items():
            low_pct, max_pct = ReviewLabelsExpectedPct[review_label].value
            annots_pct = len(review_label_annots) / total_annots
            expected_annots_pct = max(min(annots_pct, max_pct), low_pct)
            expected_annots_count = round(annots_target_count * expected_annots_pct)
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
        # dest_annots contains the original and the augmented annotations because
        # after each iteration of augmented annotations generation, the augmented 
        # annotations are saved to the destination folder along with the orignal 
        # annotation that the augmented relate to.
        src_annots, dest_annots = self._load_annots()
        all_annots = {k: v + dest_annots.get(k, []) 
                      for k, v in src_annots.items()}
        self._log_current_annots_count(all_annots)
        augmentation_infos = self._get_expected_annots_counts(all_annots, annot_target_count)

        for review_label in src_annots:
            annots = dest_annots[review_label] \
                if review_label in dest_annots and len(dest_annots[review_label]) > 0 \
                else src_annots[review_label]
            if len(annots) > augmentation_infos[review_label].total_augmentations:
                original_annots = [a for a in annots if "originated_from" not in a]
                target_augmentations_cnt = augmentation_infos[review_label].total_augmentations - len(original_annots)
                augmented_annots = sample([a for a in annots if "originated_from" in a], k=target_augmentations_cnt) \
                    if target_augmentations_cnt > 0 else []
                balanced_annots[review_label] = original_annots + augmented_annots
                self._save_annots(balanced_annots[review_label])
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
                                                   annots_to_generate)
                    balanced_annots[review_label].extend(temp_annots)
                    generated_annots_count += len(temp_annots)
                    annots_to_generate -= generated_annots_count
                        
            else:
                self._save_annots(annots)
                balanced_annots[review_label] = annots

        return CuadAnnots(balanced_annots)
