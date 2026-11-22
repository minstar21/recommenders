"""Offline data-contract checks for CPU news quickstarts."""

import pickle

import numpy as np
import pytest

from recommenders.models.newsrec.quickstart import (
    make_synthetic_news,
    require_news_files,
)


def test_refuses_to_overwrite_prepared_inputs(tmp_path):
    original = tmp_path / "news.tsv"
    original.write_text("user input", encoding="utf-8")
    with pytest.raises(ValueError, match="empty directory"):
        make_synthetic_news(tmp_path)
    assert original.read_text(encoding="utf-8") == "user input"
    assert list(tmp_path.iterdir()) == [original]


def test_missing_prepared_input_has_no_download_or_write(tmp_path):
    with pytest.raises(FileNotFoundError, match="news.tsv"):
        require_news_files(tmp_path / "absent", ["news.tsv"])
    assert not list(tmp_path.iterdir())


def test_impressions_and_vocabularies_are_consistent(tmp_path):
    root = make_synthetic_news(tmp_path)
    with (root / "utils/word_dict.pkl").open("rb") as stream:
        words = pickle.load(stream)
    embedding = np.load(root / "utils/embedding.npy")
    assert max(words.values()) < embedding.shape[0]
    split_ids = []
    for split in ("train", "valid"):
        catalog = {line.split("\t")[0] for line in (root / split / "news.tsv").read_text().splitlines()}
        ids = set()
        for line in (root / split / "behaviors.tsv").read_text().splitlines():
            imp, _, _, history, candidates = line.split("\t")
            ids.add(imp)
            labeled = [x.rsplit("-", 1) for x in candidates.split()]
            assert sum(int(label) for _, label in labeled) == 1
            assert len(labeled) == 5
            assert set(history.split()).isdisjoint(n for n, _ in labeled)
            assert set(history.split()) | {n for n, _ in labeled} <= catalog
        split_ids.append(ids)
    assert split_ids[0].isdisjoint(split_ids[1])


def test_dkn_entity_and_word_indices_fit_embeddings(tmp_path):
    root = make_synthetic_news(tmp_path) / "mind-demo-dkn"
    for line in (root / "doc_feature.txt").read_text().splitlines():
        _, words, entities = line.split()
        word_ids = list(map(int, words.split(",")))
        entity_ids = list(map(int, entities.split(",")))
        assert len(word_ids) == len(entity_ids) == 8
        assert max(word_ids) < np.load(root / "word_embeddings_100.npy").shape[0]
        assert max(entity_ids) < np.load(root / "TransE_entity2vec_100.npy").shape[0]
