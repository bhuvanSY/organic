"""
Dataset ingestion engine supporting CSV, TSV, TXT, and Parquet formats.
Implements streaming chunked reading and configurable column mapping.
"""

import csv
import os
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union
import pyarrow.parquet as pq
import pandas as pd

from chemrxn.data.schemas import ColumnMapping


def auto_detect_mapping(columns: List[str]) -> ColumnMapping:
    """
    Infers column mapping from available column names.
    Supports Lowe US patents, Jin USPTO, and standard reaction tables.
    """
    col_lower = {c.lower(): c for c in columns}
    mapping = ColumnMapping()

    # 1. Full reaction SMILES column
    for candidate in ["canonicalizedreaction", "reaction_smiles", "reactionsmiles", "reaction", "rxn"]:
        if candidate in col_lower:
            mapping.reaction_col = col_lower[candidate]
            break

    # 2. Reactants column
    for candidate in ["reactants_smiles", "reactants", "source", "reactant"]:
        if candidate in col_lower:
            mapping.reactants_col = col_lower[candidate]
            break

    # 3. Products column
    for candidate in ["products_smiles", "products", "target", "product"]:
        if candidate in col_lower:
            mapping.products_col = col_lower[candidate]
            break

    # 4. Reagents column
    for candidate in ["reagents_smiles", "reagents", "reagent"]:
        if candidate in col_lower:
            mapping.reagents_col = col_lower[candidate]
            break

    # 5. ID column
    for candidate in ["patentnumber", "patent_number", "id", "reaction_id", "rxn_id"]:
        if candidate in col_lower:
            mapping.id_col = col_lower[candidate]
            break

    # 6. Condition columns
    condition_keys = ["year", "paragraphnum", "textminedyield", "calculatedyield", "temperature", "solvent"]
    for k in condition_keys:
        if k in col_lower and col_lower[k] not in (mapping.reaction_col, mapping.reactants_col, mapping.products_col, mapping.id_col):
            mapping.condition_cols.append(col_lower[k])

    return mapping


class DatasetIngestion:
    """
    Streaming chunked ingestion engine for large chemical reaction datasets.
    Handles CSV, TSV, TXT (Jin USPTO), and Parquet without memory blowup.
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        column_mapping: Optional[ColumnMapping] = None,
        chunk_size: int = 5000,
        sample_limit: Optional[int] = None,
    ):
        p = Path(filepath)
        if not p.exists():
            # Check parent directory or package root if invoked from a subdirectory (e.g. frontend/)
            candidates = [
                Path("..") / filepath,
                Path(__file__).resolve().parent.parent.parent / filepath,
            ]
            for candidate in candidates:
                if candidate.exists():
                    p = candidate.resolve()
                    break

        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        self.filepath = p

        self.column_mapping = column_mapping
        self.chunk_size = chunk_size
        self.sample_limit = sample_limit
        self.format = self._detect_format()

    def _detect_format(self) -> str:
        ext = self.filepath.suffix.lower()
        if ext == ".parquet":
            return "parquet"
        if ext in (".tsv", ".txt") and "jin" in self.filepath.name.lower():
            return "jin_txt"
        return "delimited"  # CSV or TSV

    def stream_raw_records(self) -> Generator[Dict[str, Any], None, None]:
        """
        Yields raw records as dictionaries containing either full reaction SMILES
        or separate reactants, products, reagents, and conditions.
        """
        if self.format == "parquet":
            yield from self._stream_parquet()
        elif self.format == "jin_txt":
            yield from self._stream_jin_txt()
        else:
            yield from self._stream_delimited()

    def _stream_parquet(self) -> Generator[Dict[str, Any], None, None]:
        parquet_file = pq.ParquetFile(str(self.filepath))
        mapping = self.column_mapping or auto_detect_mapping(parquet_file.schema.names)

        count = 0
        for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                record = self._extract_record_from_row(row, mapping)
                yield record
                count += 1
                if self.sample_limit is not None and count >= self.sample_limit:
                    return

    def _stream_jin_txt(self) -> Generator[Dict[str, Any], None, None]:
        count = 0
        with open(self.filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Format: reactants>>product [bond_changes]
                record = {
                    "raw_reaction_string": line,
                    "reaction_id": f"jin_{line_idx}",
                    "conditions": {},
                }
                yield record
                count += 1
                if self.sample_limit is not None and count >= self.sample_limit:
                    return

    def _stream_delimited(self) -> Generator[Dict[str, Any], None, None]:
        """Handles comma or tab delimited files with optional comment headers (#)."""
        count = 0
        with open(self.filepath, "r", encoding="utf-8", errors="ignore") as f:
            # Skip comment lines to find header
            pos = f.tell()
            header_line = f.readline()
            while header_line and header_line.startswith("#"):
                pos = f.tell()
                header_line = f.readline()

            if not header_line:
                return

            delimiter = "\t" if "\t" in header_line else ","
            headers = [h.strip() for h in header_line.strip().split(delimiter)]
            mapping = self.column_mapping or auto_detect_mapping(headers)

            reader = csv.DictReader(f, fieldnames=headers, delimiter=delimiter)
            for row in reader:
                # Skip secondary comment headers if present
                first_val = next(iter(row.values()), "")
                if first_val and str(first_val).startswith("#"):
                    continue

                record = self._extract_record_from_dict(row, mapping, count)
                yield record
                count += 1
                if self.sample_limit is not None and count >= self.sample_limit:
                    return

    def _extract_record_from_row(self, row: pd.Series, mapping: ColumnMapping) -> Dict[str, Any]:
        return self._extract_record_from_dict(row.to_dict(), mapping, 0)

    def _extract_record_from_dict(
        self, row: Dict[str, Any], mapping: ColumnMapping, fallback_id: int
    ) -> Dict[str, Any]:
        rxn_str = row.get(mapping.reaction_col) if mapping.reaction_col else None
        reactants = row.get(mapping.reactants_col) if mapping.reactants_col else None
        products = row.get(mapping.products_col) if mapping.products_col else None
        reagents = row.get(mapping.reagents_col) if mapping.reagents_col else None
        rxn_id = str(row.get(mapping.id_col)) if mapping.id_col and row.get(mapping.id_col) else f"row_{fallback_id}"

        # If reactants/products have spaced tokens (e.g. Lowe dataset 'C S ( = O ) ...'), remove spaces
        if reactants and isinstance(reactants, str) and " " in reactants:
            reactants = reactants.replace(" ", "")
        if products and isinstance(products, str) and " " in products:
            products = products.replace(" ", "")

        # Extract extra conditions
        conditions = {}
        for c in mapping.condition_cols:
            if c in row and row[c] is not None and str(row[c]).strip() != "":
                conditions[c] = str(row[c]).strip()

        return {
            "raw_reaction_string": rxn_str,
            "reactants": reactants,
            "products": products,
            "reagents": reagents,
            "reaction_id": rxn_id,
            "conditions": conditions,
        }
