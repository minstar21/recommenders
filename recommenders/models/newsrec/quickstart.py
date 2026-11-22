# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.
"""Small, explicitly synthetic inputs for offline news-model quickstarts.

These random embeddings and invented impressions exercise code paths, not MIND
quality. No data is downloaded. Never overwrite a user's prepared input folder.
"""

from pathlib import Path
import pickle

import numpy as np
import yaml


def make_synthetic_news(root):
    """Create 32 invented articles, 8 users and disjoint train/dev impressions.

    Call in a new temporary directory. YAML dimensions are reduced for CPU use;
    all four NewsRec architectures and DKN entity/context channels remain active.
    Pickles generated here are trusted local dictionaries. External prepared
    dictionaries are Python pickles and must come from a trusted source.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("Synthetic news output must be an empty directory")
    rng = np.random.default_rng(42)
    words = {f"word{i}": i for i in range(1, 65)}
    users = {f"U{i}": i for i in range(8)}
    utils = root / "utils"
    utils.mkdir()
    for name, value in {
        "word_dict.pkl": words, "word_dict_all.pkl": words,
        "uid2index.pkl": users, "vert_dict.pkl": {"topic": 1},
        "subvert_dict.pkl": {"section": 1},
    }.items():
        with (utils / name).open("wb") as stream:
            pickle.dump(value, stream)
    embedding = rng.normal(0, 0.1, (65, 16)).astype("float32")
    embedding[0] = 0
    for name in ("embedding.npy", "embedding_all.npy"):
        np.save(utils / name, embedding)
    articles = []
    for i in range(1, 33):
        title = " ".join(f"word{j}" for j in rng.integers(1, 65, 8))
        articles.append(f"N{i}\ttopic\tsection\t{title}\t{title}\thttps://example.invalid/{i}\t[]\t[]\n")
    for split, count, offset in (("train", 32, 0), ("valid", 16, 1000)):
        folder = root / split
        folder.mkdir()
        (folder / "news.tsv").write_text("".join(articles), encoding="utf-8")
        rows = []
        for i in range(count):
            ids = rng.choice(np.arange(9, 33), size=5, replace=False)
            impressions = " ".join(f"N{n}-{int(j == 0)}" for j, n in enumerate(ids))
            rows.append(f"{i+offset}\tU{i%8}\t11/14/2019 9:00:00 AM\tN1 N2 N3 N4\t{impressions}\n")
        (folder / "behaviors.tsv").write_text("".join(rows), encoding="utf-8")
    for model in ("nrms", "lstur", "naml", "npa"):
        config = dict(
            model_type=model, data_format="naml" if model == "naml" else "news",
            title_size=8, body_size=8, his_size=4, npratio=4,
            word_emb_dim=16, head_num=2, head_dim=8, attention_hidden_dim=16,
            filter_num=16, window_size=3, gru_unit=16, type="ini",
            user_emb_dim=8, vert_emb_dim=8, subvert_emb_dim=8,
            vert_num=2, subvert_num=2, cnn_activation="relu", dense_activation="relu",
            loss="cross_entropy_loss", dropout=0.0, learning_rate=0.001,
            metrics=["group_auc", "mean_mrr", "ndcg@5;10"],
            support_quick_scoring=model != "npa", show_step=100,
        )
        (utils / f"{model}.yaml").write_text(yaml.safe_dump({"config": config}), encoding="utf-8")
    # DKN uses indexed article features and fixed per-user histories rather than TSV.
    dkn = root / "mind-demo-dkn"
    dkn.mkdir()
    features = []
    for i in range(1, 33):
        word_ids = ",".join(map(str, rng.integers(1, 65, 8)))
        entity_ids = ",".join(map(str, rng.integers(1, 17, 8)))
        features.append(f"N{i} {word_ids} {entity_ids}\n")
    (dkn / "doc_feature.txt").write_text("".join(features), encoding="utf-8")
    (dkn / "user_history.txt").write_text("".join(f"U{i} N1,N2,N3,N4\n" for i in range(8)), encoding="utf-8")
    for split, count, offset in (("train", 32, 0), ("valid", 16, 1000), ("test", 16, 2000)):
        rows = []
        for i in range(count):
            for j, n in enumerate(rng.choice(np.arange(9, 33), 5, replace=False)):
                rows.append(f"{int(j == 0)} U{i%8} N{n}%{i+offset}\n")
        (dkn / f"{split}_mind_demo.txt").write_text("".join(rows), encoding="utf-8")
    for name, shape in (("word_embeddings_100.npy", (65, 16)),
                        ("TransE_entity2vec_100.npy", (17, 16)),
                        ("TransE_context2vec_100.npy", (17, 16))):
        array = rng.normal(0, 0.1, shape).astype("float32")
        array[0] = 0
        np.save(dkn / name, array)
    config = dict(
        model_type="dkn", data_format="dkn", doc_size=8, history_size=4,
        word_size=65, entity_size=17, dim=16, entity_dim=16,
        use_entity=True, use_context=True, transform=True, method="classification",
        activation=["sigmoid"], layer_sizes=[16], dropout=[0.0],
        filter_sizes=[1, 2, 3], num_filters=8, attention_layer_sizes=8,
        attention_activation="relu", attention_dropout=0.0,
        loss="log_loss", metrics=["auc"],
        pairwise_metrics=["group_auc", "mean_mrr", "ndcg@5;10"],
        save_model=False, show_step=100,
    )
    (dkn / "dkn.yaml").write_text(yaml.safe_dump({"config": config}), encoding="utf-8")
    return root


def require_news_files(root, relative_paths):
    """Fail before training, without creating/downloading missing user inputs."""
    root = Path(root).expanduser().resolve()
    missing = [str(p) for p in relative_paths if not (root / p).is_file()]
    if missing:
        raise FileNotFoundError("Prepared news input is incomplete: " + ", ".join(missing))
    return str(root)
