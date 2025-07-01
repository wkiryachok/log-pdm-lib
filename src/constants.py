KEY_WORDS_SQL = [
    "select.+?from",
    "select.+?as",
    "select.+?union",
    r"create\s+?table",
    r"create\s+?view",
    r"create\s+?local\s+?temp",
    r"create\s+?projection",
    r"create\s+?role",
    r"create\s+?sequence",
    r"create\s+?schema",
    r"create\s+?or\s+?replace\s+?view",
    r"creating\s+?default\s+?projection",
    r"drop\s+?table",
    r"drop\s+?view",
    r"drop\s+?sequence",
    "grant.+?to",
    "grant.+?on",
    r"comment\s+?on\s+?table",
    r"comment\s+?on\s+?column",
    r"drop_partitions\s+?on\s+?table",
    r"alter\s+?table",
    r"alter\s+?session",
    r"alter\s+?sequence",
    r"alter\s+?view",
    r"alter\s+?schema",
    "update.+?set",
    r"delete\s+?from",
    r"insert\s+?into",
    r"truncate\s+?table",
    "with.+?as",
    "copy.+?from",
    "copy.+?as",
    r"copy\s+?table",
    r"replicate\s+?table",
    "merge.+?into",
    r"check\s+?owner\s+?permissions\s+?on",
    r"/\*",
]
KEY_WORDS_SQL = "|".join(KEY_WORDS_SQL)
REGEX_SQL = f"(?:{KEY_WORDS_SQL})"

TAGS_LIST_HDFS = ["INFO", "WARN"]  # HDFS
KEY_WORDS_TAGS = "|".join(TAGS_LIST_HDFS)
REGEX_TAGS_HDFS = f"(?:{KEY_WORDS_TAGS})"

DEFAULT_NLP_MODEL_PROP = {
    "nlp_model_name": "fasttext",
    "vector_size": 100,
    "window": 5,
    "min_count": 1,
    "epochs": 5,
    "n_workers": 4,
}

DEFAULT_PD_MODEL_PROP = {
    "model_name": "knn",
    "k_nearest_nbrs": 7,
}

N_WORKERS = 4
