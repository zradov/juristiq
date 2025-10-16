from typing import List
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Paraphraser:

    def __init__(
        self,
        model_name: str = "Vamsi/T5_Paraphrase_Paws",
        max_new_tokens: int = 256,
        top_k: int = 50,
        top_p: float = 0.95,
        temperature: float = 1.0,
    ):
        """
        Initializes a new instance of the Paraphraser class.

        Args:
            model_name: a name of the text generation model.
            max_length: 
        """

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


    def _tokenize(self, input_text: str) -> str:
        return self.tokenizer(
            [input_text],
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True
        )


    def _generate(self,
                  text: str, 
                  max_sentences: int, 
                  max_new_tokens: int):
        
        encoding = self._tokenize(text)
        input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=max_sentences,
            do_sample=True,
        )

        return outputs
    

    def _get_paragraphs(self, outputs: Tensor) -> list[str]:
        
        paragraphs = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        
        return paragraphs


    def _pad_paragraphs(self, 
                        paragraphs: list[str], 
                        input_text: str) -> list[str]:
        padded_paragraphs = []

        for paragraph in paragraphs:
            # if paragraph is not ending with "." it means that the sentence is cut 
            # off and it needs to be padded with using the input text.
            if paragraph[-1] != ".":
                new_paragraph = paragraph

                # try to find the same paragraph's suffix in the input text
                # and pad the paragraph with the suffix characters from the
                # input text. 
                for chunk_size in [20, 15, 10]:
                    max_last_chars = min(len(paragraph), chunk_size)
                    paragraph_suffix = paragraph[-max_last_chars:]
                    index = input_text.rfind(paragraph_suffix)
                    if index != -1:
                        new_paragraph = paragraph[:-max_last_chars] + input_text[index:]
                        break
 
                padded_paragraphs.append(new_paragraph)      

        return padded_paragraphs          
                
    
    def rephrase(self, 
                 text: str, 
                 max_sentences: int=5,
                 use_text_length: bool=False) -> List[str]:
        max_new_tokens = int(len(text)*1.5) + 1 if use_text_length else self.max_new_tokens 
        prompt_text = f"rephrase: {text}</s>"

        outputs = self._generate(prompt_text, max_sentences=max_sentences, max_new_tokens=max_new_tokens)
 
        new_paragraphs = self._get_paragraphs(outputs)
        padded_paragraphs = self._pad_paragraphs(new_paragraphs, text)
        # Filter out paragraphs that are not ending with "." character
        padded_paragraphs = [p for p in padded_paragraphs if p[-1] == "."]

        return new_paragraphs
