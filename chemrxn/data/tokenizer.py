"""
Chemical SMILES Tokenizer for organic reaction prediction.
Uses regex pattern recognition for multi-character chemical tokens,
bracketed atoms, bond symbols, stereochemistry, and special tokens.
"""

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union


SMILES_REGEX_PATTERN = r"(\[[^\]]+]|Br?|Cl?|Si?|Na?|Li?|Mg?|Al?|K?|Ca?|B|C|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"


class SMILESTokenizer:
    """
    Regex-based Chemical SMILES Tokenizer with special tokens support.
    """

    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        self.regex = re.compile(SMILES_REGEX_PATTERN)
        if vocab is not None:
            self.token_to_id = dict(vocab)
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        else:
            self.token_to_id = {token: idx for idx, token in enumerate(self.SPECIAL_TOKENS)}
            self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def sos_id(self) -> int:
        return self.token_to_id[self.SOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def tokenize(self, smiles: str) -> List[str]:
        """
        Tokenizes a SMILES string into chemical tokens.
        """
        if not smiles or not isinstance(smiles, str):
            return []
        tokens = self.regex.findall(smiles.strip())
        return tokens

    def build_vocab(self, smiles_list: Iterable[str], min_frequency: int = 1) -> None:
        """
        Builds vocabulary from an iterable of SMILES strings.
        """
        frequency: Dict[str, int] = {}
        for s in smiles_list:
            for tok in self.tokenize(s):
                frequency[tok] = frequency.get(tok, 0) + 1

        # Preserve special tokens at initial IDs
        self.token_to_id = {token: idx for idx, token in enumerate(self.SPECIAL_TOKENS)}

        # Add sorted chemical tokens based on frequency
        sorted_tokens = sorted(
            [tok for tok, cnt in frequency.items() if cnt >= min_frequency and tok not in self.SPECIAL_TOKENS]
        )

        for tok in sorted_tokens:
            new_id = len(self.token_to_id)
            self.token_to_id[tok] = new_id

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def encode(
        self,
        smiles: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """
        Encodes a SMILES string to token integer IDs.
        """
        tokens = self.tokenize(smiles)
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in tokens]

        if add_special_tokens:
            ids = [self.sos_id] + ids + [self.eos_id]

        if max_length is not None:
            if len(ids) > max_length:
                # Keep eos if present
                ids = ids[: max_length - 1] + [self.eos_id]
            elif len(ids) < max_length:
                ids = ids + [self.pad_id] * (max_length - len(ids))

        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes a list of token IDs back into a SMILES string.
        """
        tokens = []
        for tid in token_ids:
            if tid in self.id_to_token:
                tok = self.id_to_token[tid]
                if skip_special_tokens and tok in self.SPECIAL_TOKENS:
                    if tok == self.EOS_TOKEN:
                        break
                    continue
                tokens.append(tok)
            else:
                if not skip_special_tokens:
                    tokens.append(self.UNK_TOKEN)
        return "".join(tokens)

    def save_vocab(self, filepath: Union[str, Path]) -> None:
        """
        Saves vocabulary mapping to a JSON file.
        """
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, indent=2)

    @classmethod
    def load_vocab(cls, filepath: Union[str, Path]) -> "SMILESTokenizer":
        """
        Loads vocabulary mapping from a JSON file.
        """
        p = Path(filepath)
        with open(p, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab)
