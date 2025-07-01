import os
import sys
import re
import json
import gc
import gzip
import pickle
import time
import logging
from io import StringIO
from multiprocessing import Pool
from functools import partial
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from slg import SemanticLogGraphModel
from tools import read_logs
from vectorizer import LogVectorizer
from sklearn_models import KnnModel, SvmModel, IsoModel, EnvModel, LofModel
from autoencoder_model import AEModel
from constants import TAGS_LIST_HDFS, REGEX_SQL, DEFAULT_NLP_MODEL_PROP, DEFAULT_PD_MODEL_PROP, N_WORKERS


class LogAnalysis:
    """Class with logs analysis model"""

    def __init__(
        self,
        download_model: bool = False,
        path_to_model: str = "",
        path_to_semantic_model: str = "",
        use_words_tfidf: bool = True,
        use_logs_tfidf: bool = True,
        use_logs_patterns: bool = False,
        time_interval: int = 1,
        nlp_model_properties: dict = DEFAULT_NLP_MODEL_PROP,
        pd_model_properties: dict = DEFAULT_PD_MODEL_PROP,
        tags_weights: Optional[dict] = None,
        patterns_weights: Optional[dict] = None,
        quantile: Optional[float] = 0.95,
        save_train_info: Optional[bool] = True,
        log_file: str = None,
        log_analysis_model_name: Optional[str] = "log_analysis_model.json",
        pd_model_name: Optional[str] = "pd_model.pickle",
        doc_vectors_name: Optional[str] = "doc_vectors.csv",
    ):
        self.download_model = download_model
        self.path_to_model = path_to_model
        self.path_to_semantic_model = path_to_semantic_model
        self.log_analysis_model_name = log_analysis_model_name
        self.log_file = log_file
        self.__setup_logger(log_file)

        if download_model:
            log_analysis_model_path = os.path.join(
                path_to_model, self.log_analysis_model_name
            )
            self.logger.info(
                "Loading log analysis model from: %s", log_analysis_model_path
            )
            with open(
                log_analysis_model_path, mode="r", encoding="utf-8"
            ) as opened_file:
                log_analysis_model = json.load(opened_file, parse_int=int)

            self.use_words_tfidf = log_analysis_model.get("use_words_tfidf", True)
            self.use_logs_tfidf = log_analysis_model.get("use_logs_tfidf", False)
            self.use_logs_patterns = log_analysis_model.get("use_logs_patterns", False)
            self.time_interval = log_analysis_model.get("time_interval", 1)
            self.nlp_model_properties = log_analysis_model.get(
                "nlp_model_properties",
                DEFAULT_NLP_MODEL_PROP
            )
            self.pd_model_properties = log_analysis_model.get(
                "pd_model_properties",
                DEFAULT_PD_MODEL_PROP
            )
            self.tags_weights = log_analysis_model.get("tags_weights", None)
            self.patterns_weights = log_analysis_model.get("patterns_weights", None)
            self.quantile = log_analysis_model.get("quantile", 0.95)
            self.save_train_info = log_analysis_model.get("save_train_info", True)
            self.anomaly_score = pd.read_json(
                StringIO(log_analysis_model.get("anomaly_score", None)), typ="series"
            )
            self.threshold = log_analysis_model.get("threshold", 0)
            self.fitted = log_analysis_model.get("fitted", True)
            self.pd_model_name = log_analysis_model.get(
                "pd_model_name", "pd_model.pickle"
            )
            self.doc_vectors_name = log_analysis_model.get(
                "doc_vectors_name", "doc_vectors.csv"
            )
        else:
            if os.path.exists(
                os.path.join(self.path_to_model, self.log_analysis_model_name)
            ):
                self.logger.info(
                    "Log analysis model has already created. "
                    "This model will be rewritten."
                )

            self.use_words_tfidf = use_words_tfidf
            self.use_logs_tfidf = use_logs_tfidf
            self.use_logs_patterns = use_logs_patterns
            self.time_interval = time_interval
            self.nlp_model_properties = nlp_model_properties
            self.pd_model_properties = pd_model_properties
            self.tags_weights = tags_weights
            self.patterns_weights = patterns_weights
            self.quantile = quantile
            self.save_train_info = save_train_info
            self.train_anomaly_score = None
            self.threshold = None
            self.fitted = False
            self.pd_model_name = pd_model_name
            self.doc_vectors_name = doc_vectors_name

        self.logger.info("Model parameter 'download_model' = %s", self.download_model)
        self.logger.info("Model parameter 'path_to_model' = %s", self.path_to_model)
        self.logger.info(
            "Model parameter 'log_analysis_model_name' = %s", log_analysis_model_name
        )
        self.logger.info(
            "Model parameter 'path_to_semantic_model' = %s", self.path_to_semantic_model
        )
        self.logger.info("Model parameter 'use_words_tfidf' = %s", self.use_words_tfidf)
        self.logger.info("Model parameter 'use_logs_tfidf' = %s", self.use_logs_tfidf)
        self.logger.info(
            "Model parameter 'use_logs_patterns' = %s", self.use_logs_patterns
        )
        self.logger.info("Model parameter 'time_interval' = %s", self.time_interval)
        self.logger.info("Model parameter 'nlp_model_properties' = %s", self.nlp_model_properties)
        self.logger.info("Model parameter 'pd_model_properties' = %s", self.pd_model_properties)
        self.logger.info("Model parameter 'tags_weights' = %s", self.tags_weights)
        self.logger.info(
            "Model parameter 'patterns_weights' = %s", self.patterns_weights
        )
        self.logger.info(
            "Model parameter 'quantile' = %s", self.quantile
        )
        self.logger.info("Model parameter 'save_train_info' = %s", self.save_train_info)
        self.logger.info(
            "Model parameter 'doc_vectors_name' = %s", self.doc_vectors_name
        )
        if download_model:
            self.logger.info(
                "Model parameter 'threshold' = %s", self.threshold
            )
            self.logger.info("Model parameter 'fitted' = %s", self.fitted)

        if use_logs_patterns & (tags_weights is None) & (patterns_weights is None):
            self.logger.error(
                "tags_weights or patterns_weights must not be None if "
                "use_logs_patterns is True"
            )
            raise ValueError(
                "tags_weights or patterns_weights must not be None if "
                "use_logs_patterns is True"
            )

    def __setup_logger(self, log_file: str) -> None:
        self.logger = logging.getLogger("LogAnalysis")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] <%(levelname)s> - %(message)s"
        )

        ch = logging.StreamHandler(sys.stdout)  # console
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if log_file is not None:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def __log_preprocessing(
        self, logs: list, dataset_type: str, delete_sql: bool = False
    ) -> Tuple[pd.DataFrame, list]:
        """Log preprocessing function"""
        with open(self.path_to_semantic_model, "rb") as handle:
            slg_model = pickle.load(handle)

        if dataset_type == "HDFS":
            tags_list = TAGS_LIST_HDFS
        else:
            ## TODO: add other types of datasets
            pass

        log_pattern_list = []
        ts_list = []
        tag_list = []
        for raw_log in tqdm(logs):
            if delete_sql:
                search_res = re.search(REGEX_SQL, re.sub("\n", " ", raw_log), re.IGNORECASE)
                if search_res is not None:
                    raw_log = raw_log[: search_res.start()]
            ts, log = slg_model.preprocess_logline_tokens(raw_log)
            for i, word in enumerate(log):
                if word in tags_list:
                    if "window" in self.nlp_model_properties:
                        if (len(log) - i - 1) >= self.nlp_model_properties["window"]:
                            log_pattern_list.append(" ".join(log))
                            ts_list.append(ts)
                            tag_list.append(word)
                        break
                    else:
                        log_pattern_list.append(" ".join(log))
                        ts_list.append(ts)
                        tag_list.append(word)                        
        timestamp = np.array(ts_list, dtype="datetime64[s]")

        logs_df = pd.DataFrame(index=timestamp)
        logs_df["log_pattern"] = log_pattern_list
        logs_df["tag"] = tag_list
        return logs_df

    def __sorting_check(self, logs_df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        timestamp = logs_df.index.to_numpy()
        if not all(timestamp[i] <= timestamp[i + 1] for i in range(len(timestamp) - 1)):
            self.logger.warning("Dataset is not sorted, start sorting the dataset")
            sort_indexes = np.argsort(timestamp)
            logs_df = logs_df.iloc[sort_indexes]
        return logs_df

    def __split_list(self, lst: list) -> list:
        """Splitting list to n_workers parts function"""
        n = int(len(lst) / N_WORKERS)
        return [lst[i : i + n] for i in range(0, len(lst), n)]

    def __parallel_log_preprocessing(self, logs: list) -> Tuple[pd.DataFrame, list]:
        """Parallel log preprocessing function"""
        splited_logs = self.__split_list(logs)
        with Pool(N_WORKERS) as pool:
            logs_df = pool.map(
                partial(self.__log_preprocessing, dataset_type="HDFS"), splited_logs
            )
        logs_df = pd.concat(logs_df)
        return logs_df

    def __calc_anomaly_score(
        self, doc_vectors: pd.DataFrame, mode: str = "train"
    ) -> pd.Series:
        if mode == "train":
            model_name = self.pd_model_properties.pop("model_name")
            if model_name == "knn":
                model = KnnModel(self.pd_model_properties)
            elif model_name == "svm":
                model = SvmModel(self.pd_model_properties)
            elif model_name == "iso":
                model = IsoModel(self.pd_model_properties)
            elif model_name == "env":
                model = EnvModel(self.pd_model_properties)
            elif model_name == "lof":
                model = LofModel(self.pd_model_properties)
            elif model_name == "ae":
                model = AEModel(self.pd_model_properties)
            else:
                self.logger.error(
                    "Unknown pd model name: %s",
                    self.pd_model_properties["model_name"],
                )  
                raise ValueError(
                    "Unknown pd model name: %s",
                    self.pd_model_properties["model_name"],
                )              
            self.pd_model_properties["model_name"] = model_name

            self.logger.info(
                "Saving pd model to: %s",
                os.path.join(self.path_to_model, self.pd_model_name),
            )
            pd_model_pickle = open(
                os.path.join(self.path_to_model, self.pd_model_name), "wb"
            )
            pickle.dump(model, pd_model_pickle)
            pd_model_pickle.close()
        else:
            self.logger.info(
                "Loading pd model from: %s",
                os.path.join(self.path_to_model, self.pd_model_name),
            )
            model = pickle.load(
                open(os.path.join(self.path_to_model, self.pd_model_name), "rb")
            )
            anomaly_score = model.predict(doc_vectors)
        return anomaly_score

    def fit(
        self,
        path_to_files: Optional[list] = None,
        path_to_pretrained_nlp_model: Optional[str] = None,
        skip_fitting_nlp_model: Optional[bool] = False,
        skip_calc_vectors: Optional[bool] = False,
        parallel_mode: Optional[bool] = False,
    ) -> None:
        """Fit function"""
        if (
            (path_to_files is None)
            & (skip_fitting_nlp_model is False)
            & (skip_calc_vectors is False)
        ):
            self.logger.error(
                "skip_fitting_nlp_model or skip_calc_vectors must be True "
                "if path_to_data or files is None"
            )
            raise ValueError(
                "skip_fitting_nlp_model or skip_calc_vectors must be True "
                "if path_to_data or files is None"
            )
        if (path_to_files is not None) & (
            (skip_fitting_nlp_model is True) | (skip_calc_vectors is True)
        ):
            self.logger.error(
                "skip_fitting_nlp_model or skip_calc_vectors must be False "
                "if path_to_data or files is not None"
            )
            raise ValueError(
                "skip_fitting_nlp_model or skip_calc_vectors must be False "
                "if path_to_data or files is not None"
            )
        if (skip_fitting_nlp_model is True) & (skip_calc_vectors is True):
            self.logger.error(
                "skip_fitting_nlp_model and skip_calc_vectors cannot be True "
                "at the same time"
            )
            raise ValueError(
                "skip_fitting_nlp_model and skip_calc_vectors cannot be True "
                "at the same time"
            )

        if (not skip_fitting_nlp_model) & (not skip_calc_vectors):
            for path_to_file in path_to_files:
                self.logger.info("Start training from file: %s", path_to_file)
                if len(re.findall("[0-9]", path_to_file)) != 0:
                    suffix = re.findall("[0-9]", path_to_file)[0]
                else:
                    suffix = "1"
                self.logger.info("Start reading log file: %s", path_to_file)
                logs = read_logs(path_to_file, "HDFS")  # hardcode
                self.logger.info("Start preprocessing log file: %s", path_to_file)
                start_time = time.time()
                if parallel_mode:
                    self.logger.info(
                        "Using %s workers for parallel calculation", N_WORKERS
                    )
                    logs_df = self.__parallel_log_preprocessing(logs)
                else:
                    logs_df = self.__log_preprocessing(logs, dataset_type="HDFS")
                logs_df = self.__sorting_check(logs_df)
                self.logger.info(
                    "Finished preprocessing log file, time: %s sec",
                    np.round(time.time() - start_time, 2),
                )

                if not os.path.exists(os.path.join(self.path_to_model, "nlp_model")):
                    self.logger.info("Log analysis model not found, creating new model")
                    os.makedirs(os.path.join(self.path_to_model, "train_info"))
                    os.makedirs(os.path.join(self.path_to_model, "nlp_model"))
                    os.makedirs(os.path.join(self.path_to_model, "document_frequency"))

                    # inicialize vectorizer
                    log_vectorizer = LogVectorizer(
                        path_to_pretrained_nlp_model=path_to_pretrained_nlp_model,
                        path_to_semantic_model=self.path_to_semantic_model,
                        use_words_tfidf=self.use_words_tfidf,
                        use_logs_tfidf=self.use_logs_tfidf,
                        use_logs_patterns=self.use_logs_patterns,
                        nlp_model_properties=self.nlp_model_properties,
                        tags_weights=self.tags_weights,
                        patterns_weights=self.patterns_weights,
                        time_interval=self.time_interval,
                        log_file=self.log_file,
                    )
                else:
                    self.logger.info("Using log analysis model: %s", self.path_to_model)
                    # inicialize vectorizer
                    log_vectorizer = LogVectorizer(
                        download_model=True,
                        model_path=self.path_to_model,
                        path_to_semantic_model=self.path_to_semantic_model,
                        use_words_tfidf=self.use_words_tfidf,
                        use_logs_tfidf=self.use_logs_tfidf,
                        use_logs_patterns=self.use_logs_patterns,
                        nlp_model_properties=self.nlp_model_properties,
                        tags_weights=self.tags_weights,
                        patterns_weights=self.patterns_weights,
                        time_interval=self.time_interval,
                        log_file=self.log_file,
                    )
                # fit vectorizer
                start_time = time.time()
                bol_corpus, bow_corpus, logs_split_df = log_vectorizer.fit(logs_df)
                self.logger.info(
                    "Finished log analysis model fitting, time: %s sec",
                    np.round(time.time() - start_time, 2),
                )

                self.logger.info("Model updating")

                # save nlp model
                if self.nlp_model_properties["nlp_model_name"] == "word2vec":
                    full_nlp_model_name = "word2vec.model"
                elif self.nlp_model_properties["nlp_model_name"] == "fasttext":
                    full_nlp_model_name = "fasttext.model"
                else:
                    # TODO add other vectorizers
                    pass
                nlp_path = os.path.join(
                    self.path_to_model, "nlp_model", full_nlp_model_name
                )
                log_vectorizer.nlp_model.save(nlp_path)

                # save words_idf_dict
                words_idf_dict_path = os.path.join(
                    self.path_to_model,
                    "document_frequency",
                    "words_document_frequency.gz",
                )
                with gzip.open(words_idf_dict_path, mode="wb") as opened_file:
                    opened_file.write(
                        json.dumps(log_vectorizer.words_idf_dict, indent=True).encode(
                            "utf-8"
                        )
                    )

                # save logs_idf_dict
                logs_idf_dict_path = os.path.join(
                    self.path_to_model,
                    "document_frequency",
                    "logs_document_frequency.gz",
                )
                with gzip.open(logs_idf_dict_path, mode="wb") as opened_file:
                    opened_file.write(
                        json.dumps(log_vectorizer.logs_idf_dict, indent=True).encode(
                            "utf-8"
                        )
                    )

                # save bol corpus
                bol_corpus_path = os.path.join(
                    self.path_to_model, "train_info", f"bol_corpus_{suffix}.gz"
                )
                with gzip.open(bol_corpus_path, mode="wb") as opened_file:
                    opened_file.write(
                        json.dumps(bol_corpus, indent=True).encode("utf-8")
                    )

                # save bow corpus
                bow_corpus_path = os.path.join(
                    self.path_to_model, "train_info", f"bow_corpus_{suffix}.gz"
                )
                with gzip.open(bow_corpus_path, mode="wb") as opened_file:
                    opened_file.write(
                        json.dumps(bow_corpus, indent=True).encode("utf-8")
                    )

                # save logs_split_df
                logs_split_df_path = os.path.join(
                    self.path_to_model, "train_info", f"logs_split_df_{suffix}.csv"
                )
                logs_split_df.to_csv(logs_split_df_path, encoding="utf-8")

                # documents_count
                with open(
                    os.path.join(
                        self.path_to_model, "document_frequency", "documents_count.txt"
                    ),
                    mode="w",
                    encoding="utf-8",
                ) as file:
                    file.write(str(log_vectorizer.documents_count))

                gc.collect()
                del logs, logs_df, bol_corpus, log_vectorizer, logs_split_df

        if not skip_calc_vectors:
            self.logger.info("Start train vectors calculation")
            log_vectorizer = LogVectorizer(
                download_model=True,
                model_path=self.path_to_model,
                path_to_semantic_model=self.path_to_semantic_model,
                use_words_tfidf=self.use_words_tfidf,
                use_logs_tfidf=self.use_logs_tfidf,
                use_logs_patterns=self.use_logs_patterns,
                tags_weights=self.tags_weights,
                patterns_weights=self.patterns_weights,
                log_file=self.log_file,
            )
            start_time = time.time()
            doc_vectors = log_vectorizer.calc_train_vectors(parallel_mode=parallel_mode)
            self.logger.info(
                "Finished train vectors calculation, time: %s sec",
                np.round(time.time() - start_time, 2),
            )

            if self.save_train_info:
                self.logger.info(
                    "Saving training vectors to folder: %s", self.path_to_model
                )
                doc_vectors.to_csv(
                    os.path.join(self.path_to_model, self.doc_vectors_name)
                )
            else:
                path_to_train_info = os.path.join(self.path_to_model, "train_info")
                self.logger.warning(
                    "Deleting files from folder: %s", path_to_train_info
                )

                for file in os.listdir(path_to_train_info):
                    os.remove(path_to_train_info)

                if os.path.exists(
                    os.path.join(self.path_to_model, self.doc_vectors_name)
                ):
                    self.logger.info(
                        "Updating existing file with training vectors in folder: %s",
                        self.path_to_model,
                    )
                    doc_vectors_cur = pd.read_csv(
                        os.path.join(self.path_to_model, self.doc_vectors_name),
                        index_col=0,
                    )
                    doc_vectors_cur.index = pd.to_datetime(doc_vectors_cur.index)
                    doc_vectors = pd.concat([doc_vectors_cur, doc_vectors])
                    doc_vectors.to_csv(
                        os.path.join(self.path_to_model, self.doc_vectors_name)
                    )
                else:
                    self.logger.info(
                        "Saving new training vectors to folder: %s", self.path_to_model
                    )
                    doc_vectors.to_csv(
                        os.path.join(self.path_to_model, self.doc_vectors_name)
                    )
        else:
            self.logger.info("Updating file with training vectors")
            doc_vectors = pd.read_csv(
                os.path.join(self.path_to_model, self.doc_vectors_name), index_col=0
            )
            doc_vectors.index = pd.to_datetime(doc_vectors.index)

        self.logger.info("Start calculating mean distances with KNN")
        doc_vectors.dropna(inplace=True)
        self.__calc_anomaly_score(
            doc_vectors.iloc[:, :self.nlp_model_properties["vector_size"]], mode="train"
        )
        self.train_anomaly_score = self.__calc_anomaly_score(
            doc_vectors.iloc[:, :self.nlp_model_properties["vector_size"]], mode="test"
        )
        self.threshold = np.quantile(self.train_anomaly_score, self.quantile)
        self.fitted = True
        self.save_model()

    def get_train_anomaly_score(self):
        """Train anomaly score getter function"""
        if self.train_anomaly_score is None:
            self.logger.error("Train anomaly score wasn't calculated")
            raise ValueError("Train anomaly score wasn't calculated")
        return self.train_anomaly_score

    def get_threshold(self):
        """Threshold getter function"""
        if self.threshold is None:
            self.logger.error("Threshold wasn't calculated")
            raise ValueError("Threshold wasn't calculated")
        return self.threshold

    def save_model(self) -> None:
        """Model saving function"""
        self.logger.info("Saving log analysis model to folder: %s", self.path_to_model)
        log_analysis_model = {
            "use_words_tfidf": self.use_words_tfidf,
            "use_logs_tfidf": self.use_logs_tfidf,
            "use_logs_patterns": self.use_logs_patterns,
            "time_interval": self.time_interval,
            "nlp_model_properties": self.nlp_model_properties,
            "pd_model_properties": self.pd_model_properties,
            "tags_weights": self.tags_weights,
            "patterns_weights": self.patterns_weights,
            "quantile": self.quantile,
            "save_train_info": self.save_train_info,
            "fitted": self.fitted,
            "pd_model_name": self.pd_model_name,
            "doc_vectors_name": self.doc_vectors_name,
            "threshold": self.threshold,
            "train_anomaly_score": self.train_anomaly_score.to_json(),
        }

        log_analysis_model_path = os.path.join(
            self.path_to_model, self.log_analysis_model_name
        )
        with open(log_analysis_model_path, mode="w", encoding="utf-8") as opened_file:
            json.dump(log_analysis_model, opened_file, indent=True)

    def predict(
        self,
        path_to_results: str,
        path_to_files: Optional[list] = None,
        skip_preprocessing: Optional[bool] = False,
        skip_calc_vectors: Optional[bool] = False,
        calc_logs_importance: Optional[bool] = False,
        save_test_info: Optional[bool] = True,
        parallel_mode: Optional[bool] = False,
    ) -> None:
        """Test data prediction function"""
        if not self.fitted:
            self.logger.error("Model not fitted")
            raise SystemError("Model not fitted")
        if (path_to_files is None) & (
            (skip_calc_vectors is False) & (skip_preprocessing is False)
        ):
            self.logger.error(
                "skip_calc_vectors or skip_preprocessing must be True if "
                "path_to_data or files is None"
            )
            raise ValueError(
                "skip_calc_vectors or skip_preprocessing must be True if "
                "path_to_data or files is None"
            )
        if (path_to_files is not None) & (
            (skip_calc_vectors is True) | (skip_preprocessing is True)
        ):
            self.logger.error(
                "skip_calc_vectors or skip_preprocessing must be False if "
                "path_to_data or files is not None"
            )
            raise ValueError(
                "skip_calc_vectors or skip_preprocessing must be False if "
                "path_to_data or files is not None"
            )
        if (skip_calc_vectors is True) & (skip_preprocessing is True):
            self.logger.error(
                "skip_calc_vectors and skip_preprocessing cannot be True "
                "at the same time"
            )
            raise ValueError(
                "skip_calc_vectors and skip_preprocessing cannot be True "
                "at the same time"
            )
        if (save_test_info is False) & (skip_preprocessing is True):
            self.logger.error(
                "save_test_info must be True when skip_preprocessing is True"
            )
            raise ValueError(
                "save_test_info must be True when skip_preprocessing is True"
            )

        if (calc_logs_importance is True) & (skip_calc_vectors is True):
            self.logger.error(
                "skip_calc_vectors must be False when calc_logs_importance is True"
            )
            raise ValueError(
                "skip_calc_vectors must be False when calc_logs_importance is True"
            )

        if (not skip_calc_vectors) & (not skip_preprocessing):
            for path_to_file in path_to_files:
                self.logger.info("Start inference for file: %s", path_to_file)

                self.logger.info("Start reading log file: %s", path_to_file)
                logs = read_logs(path_to_file, "HDFS")  # hardcode
                self.logger.info("Start preprocessing log file: %s", path_to_file)
                start_time = time.time()
                if parallel_mode:
                    self.logger.info(
                        "Using %s workers for parallel calculation", self.n_workers
                    )
                    logs_df = self.__parallel_log_preprocessing(logs)
                else:
                    logs_df = self.__log_preprocessing(logs, dataset_type="HDFS")
                logs_df = self.__sorting_check(logs_df)
                self.logger.info(
                    "Finished preprocessing log file, time: %s sec",
                    np.round(time.time() - start_time, 2),
                )

                self.logger.info("Using log analysis model: %s", self.path_to_model)
                log_vectorizer = LogVectorizer(
                    download_model=True,
                    model_path=self.path_to_model,
                    path_to_semantic_model=self.path_to_semantic_model,
                    nlp_model_name=self.nlp_model_name,
                    use_words_tfidf=self.use_words_tfidf,
                    use_logs_tfidf=self.use_logs_tfidf,
                    use_logs_patterns=self.use_logs_patterns,
                    tags_weights=self.tags_weights,
                    patterns_weights=self.patterns_weights,
                    time_interval=self.time_interval,
                    log_file=self.log_file,
                )
                self.logger.info("Start test vectors calculation")
                start_time = time.time()
                (
                    doc_vectors,
                    bol_corpus,
                    bow_corpus,
                    logs_split_df,
                    logs_importance_corpus,
                ) = log_vectorizer.calc_test_vectors(
                    logs_df,
                    calc_logs_importance=calc_logs_importance,
                    parallel_mode=parallel_mode,
                )
                self.logger.info(
                    "Finished test vectors calculation, time: %s sec",
                    np.round(time.time() - start_time, 2),
                )
                gc.collect()
                del logs_df, logs
                if not save_test_info:
                    del bol_corpus, bow_corpus

                if not os.path.exists(path_to_results):
                    self.logger.info(
                        "Results folder not found, create new results folder: %s",
                        path_to_results,
                    )
                    os.makedirs(path_to_results)
                    if save_test_info:
                        os.makedirs(os.path.join(path_to_results, "test_info"))

                if save_test_info:
                    self.logger.info(
                        "Saving bags of words and logs corpuses to folder: %s",
                        os.path.join(path_to_results, "test_info"),
                    )
                    if len(re.findall("[0-9]", path_to_file)) != 0:
                        suffix = re.findall("[0-9]", path_to_file)[0]
                    else:
                        suffix = "1"

                    # save bol corpus
                    bol_corpus_path = os.path.join(
                        path_to_results, "test_info", f"bol_corpus_{suffix}.gz"
                    )
                    with gzip.open(bol_corpus_path, mode="wb") as opened_file:
                        opened_file.write(
                            json.dumps(bol_corpus, indent=True).encode("utf-8")
                        )

                    # save bow corpus
                    bow_corpus_path = os.path.join(
                        path_to_results, "test_info", f"bow_corpus_{suffix}.gz"
                    )
                    with gzip.open(bow_corpus_path, mode="wb") as opened_file:
                        opened_file.write(
                            json.dumps(bow_corpus, indent=True).encode("utf-8")
                        )

                    # save logs_split_df
                    logs_split_df_path = os.path.join(
                        path_to_results, "test_info", f"logs_split_df_{suffix}.csv"
                    )
                    logs_split_df.to_csv(logs_split_df_path, encoding="utf-8")

                time_intervals_dict_path = os.path.join(
                    path_to_results, "time_intervals_dict.txt"
                )
                self.logger.info(
                    "Saving time intervals dictionary to: %s", time_intervals_dict_path
                )
                time_intervals_dict = {
                    os.path.basename(path_to_file): [
                        str(doc_vectors.index[0]),
                        str(doc_vectors.index[-1]),
                    ]
                }
                if os.path.exists(time_intervals_dict_path):
                    with open(
                        time_intervals_dict_path, mode="r", encoding="utf-8"
                    ) as opened_file:
                        time_intervals_dict_cur = json.load(opened_file, parse_int=int)
                    time_intervals_dict_cur.update(time_intervals_dict)
                    time_intervals_dict = time_intervals_dict_cur

                with open(
                    time_intervals_dict_path, mode="w", encoding="utf-8"
                ) as opened_file:
                    json.dump(time_intervals_dict, opened_file, indent=True)

                if os.path.exists(os.path.join(path_to_results, self.doc_vectors_name)):
                    self.logger.info(
                        "Updating file with test vectors: %s",
                        os.path.join(path_to_results, self.doc_vectors_name),
                    )
                    test_doc_vectors_cur = pd.read_csv(
                        os.path.join(path_to_results, self.doc_vectors_name),
                        index_col=0,
                    )
                    test_doc_vectors_cur.index = pd.to_datetime(
                        test_doc_vectors_cur.index
                    )

                    doc_vectors = pd.concat([test_doc_vectors_cur, doc_vectors])
                    doc_vectors.fillna(0, inplace=True)
                    doc_vectors = pd.concat(
                        [
                            doc_vectors.iloc[:, : self.vector_size],
                            doc_vectors.iloc[:, self.vector_size :].astype(int),
                        ],
                        axis=1,
                    )
                    doc_vectors.to_csv(
                        os.path.join(path_to_results, self.doc_vectors_name)
                    )
                else:
                    self.logger.info(
                        "Create new file with testing vectors: %s",
                        os.path.join(path_to_results, self.doc_vectors_name),
                    )
                    doc_vectors.to_csv(
                        os.path.join(path_to_results, self.doc_vectors_name)
                    )

        if skip_preprocessing:
            self.logger.info("Start inference with ready bags of words and logs files")
            log_vectorizer = LogVectorizer(
                download_model=True,
                model_path=self.path_to_model,
                path_to_semantic_model=self.path_to_semantic_model,
                nlp_model_name=self.nlp_model_name,
                use_words_tfidf=self.use_words_tfidf,
                use_logs_tfidf=self.use_logs_tfidf,
                use_logs_patterns=self.use_logs_patterns,
                tags_weights=self.tags_weights,
                patterns_weights=self.patterns_weights,
                time_interval=self.time_interval,
                log_file=self.log_file,
            )
            self.logger.info("Start test vectors calculation")
            start_time = time.time()
            doc_vectors, _, _, _, logs_importance_corpus = (
                log_vectorizer.calc_test_vectors(
                    path_to_results=path_to_results,
                    calc_logs_importance=calc_logs_importance,
                    parallel_mode=parallel_mode,
                )
            )
            self.logger.info(
                "Finished test vectors calculation, time: %s sec",
                np.round(time.time() - start_time, 2),
            )
            doc_vectors.to_csv(os.path.join(path_to_results, self.doc_vectors_name))

        if skip_calc_vectors:
            self.logger.info(
                "Uploading file with testing vectors from: %s",
                os.path.join(path_to_results, self.doc_vectors_name),
            )
            doc_vectors = pd.read_csv(
                os.path.join(path_to_results, self.doc_vectors_name), index_col=0
            )
            doc_vectors.index = pd.to_datetime(doc_vectors.index)

        self.logger.info("Start calculating mean distances with KNN")
        doc_vectors.dropna(inplace=True)
        mean_distances = self.__calc_mean_distances(
            doc_vectors.iloc[:, : self.vector_size], mode="test"
        )
        if calc_logs_importance:
            return mean_distances, logs_importance_corpus
        return mean_distances
