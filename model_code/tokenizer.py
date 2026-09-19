import os
import logging
from typing import List, Iterable, Optional

from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Digits
from tokenizers.normalizers import NFD, StripAccents

try:
    import pymysql
except ImportError:
    pymysql = None


def get_sql_data(
    table_name: str,
    host: str = "localhost",
    user: str = "root",
    passwd: str = "1234",
    db: str = "dockerfile_completion",
    port: int = 3306,
) -> List[str]:
    """Fetch raw dockerfile text from MySQL database."""
    if pymysql is None:
        raise ImportError("pymysql is required for database fetching. Install via 'pip install pymysql'.")
    con = pymysql.connect(host=host, user=user, passwd=passwd, db=db, port=port)
    try:
        with con.cursor() as cur:
            cur.execute(f"SELECT content_valid FROM {table_name};")
            rows = cur.fetchall()
            return [row[0].replace('\n', ' ').strip() for row in rows if row and row[0]]
    finally:
        con.close()


def build_tokenizer(vocab_size: int = 50000) -> Tokenizer:
    """Build and initialize an un-trained BPE tokenizer with normalizers and pre-tokenizers."""
    normalizer = normalizers.Sequence([
        NFD(),
        StripAccents(),
    ])
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        Whitespace(),
        Digits(individual_digits=False),
    ])
    tokenizer.normalizer = normalizer
    return tokenizer


def train_tokenizer(
    iterator: Iterable[str],
    vocab_size: int = 50000,
    output_dir: str = "tokenizer",
    min_frequency: int = 2,
) -> Tokenizer:
    """Train BPE tokenizer from an iterator of texts and save output."""
    tokenizer = build_tokenizer(vocab_size=vocab_size)
    trainer = BpeTrainer(
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        vocab_size=vocab_size,
        show_progress=True,
        min_frequency=min_frequency,
    )
    tokenizer.train_from_iterator(iterator, trainer)
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_model(output_dir)
    print(f"Trained tokenizer saved to '{output_dir}'. Vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer


if __name__ == '__main__':
    # When running directly, check if DB is configured, otherwise provide usage guidance
    db_name = os.environ.get("DOCKERFILL_DB", "dockerfile_completion")
    try:
        texts = get_sql_data("dockerfile", db=db_name)
        if texts:
            train_tokenizer(texts)
    except Exception as e:
        print(f"Could not connect to database to train tokenizer: {e}")
        print("To train from an iterator or file, call train_tokenizer(texts) in your script.")
