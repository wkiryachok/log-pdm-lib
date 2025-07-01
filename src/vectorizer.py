"""NLP vectorizer class"""

import os
import sys
import re
import json
import gc
import gzip
import time
import logging
from multiprocessing import Pool
from functools import partial
from typing import Tuple, Optional
from collections import Counter
import numpy as np
import pandas as pd
from gensim.models import FastText
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
from constants import REGEX_TAGS_HDFS


class LogVectorizer:
    """NLP vectorizer class"""

    def __init__(
        self,
        path_to_pretrained_nlp_model: str = None,
        path_to_semantic_model: str = None,
        download_model: bool = False,
        model_path: str = "",
        use_words_tfidf: bool = True,
        use_logs_tfidf: bool = True,
        use_logs_patterns: bool = False,
        nlp_model_properties: dict = None,
        tags_weights: dict = None,
        patterns_weights: dict = None,
        time_interval: float = 1,
        log_file: str = None,
    ):
        self.__setup_logger(log_file)
        self.path_to_pretrained_nlp_model = path_to_pretrained_nlp_model
        self.path_to_semantic_model = path_to_semantic_model
        self.download_model = download_model
        self.model_path = model_path
        self.use_words_tfidf = use_words_tfidf
        self.use_logs_tfidf = use_logs_tfidf
        self.use_logs_patterns = use_logs_patterns
        self.nlp_model_properties = nlp_model_properties
        self.tags_weights = tags_weights
        self.patterns_weights = patterns_weights
        self.time_interval = time_interval

        self.nlp_model_name = nlp_model_properties["nlp_model_name"]
        self.vector_size = nlp_model_properties["vector_size"]
        self.window = nlp_model_properties["window"]
        self.min_count = nlp_model_properties["min_count"]
        self.epochs = nlp_model_properties["epochs"]
        self.n_workers = nlp_model_properties["n_workers"]
        self.words_idf_dict = {}
        self.logs_idf_dict = {}
        self.words_tfidf_dict = {}
        self.logs_tfidf_dict = {}
        self.documents_count = 0

        if self.nlp_model_name not in ["word2vec", "fasttext"]:
            self.logger.error("Unknown NLP model: %s", self.nlp_model_name)
            raise ValueError(f"Unknown NLP model: {self.nlp_model_name}")

        if path_to_pretrained_nlp_model is not None:
            if "word2vec" in path_to_pretrained_nlp_model:
                self.logger.info(
                    "Loading pretrained word2vec model from: %s",
                    path_to_pretrained_nlp_model,
                )
                self.nlp_model = Word2Vec.load(path_to_pretrained_nlp_model)
            elif "fasttext" in path_to_pretrained_nlp_model:
                self.logger.info(
                    "Loading pretrained fasttext model from: %s",
                    path_to_pretrained_nlp_model,
                )
                self.nlp_model = FastText.load(path_to_pretrained_nlp_model)
            else:
                self.logger.error(
                    "Pretrained NLP model must have 'fasttext' or 'word2vec' in name"
                )
                raise ValueError(
                    "Pretrained NLP model must have 'fasttext' or 'word2vec' in name"
                )
        else:
            self.nlp_model = None

        if self.download_model:
            if self.nlp_model_name == "word2vec":
                nlp_path = os.path.join(self.model_path, "nlp_model", "word2vec.model")
                self.logger.info("Loading word2vec model from: %s", nlp_path)
                self.nlp_model = Word2Vec.load(nlp_path)

            elif self.nlp_model_name == "fasttext":
                nlp_path = os.path.join(self.model_path, "nlp_model", "fasttext.model")
                self.logger.info("Loading fasttext model from: %s", nlp_path)
                self.nlp_model = FastText.load(nlp_path)

            self.vector_size = self.nlp_model.vector_size
            self.window = self.nlp_model.window
            self.min_count = self.nlp_model.min_count

            words_idf_dict_path = os.path.join(
                self.model_path, "document_frequency", "words_document_frequency.gz"
            )
            self.logger.info(
                "Loading words idf dictionary from: %s", words_idf_dict_path
            )
            with gzip.open(words_idf_dict_path, mode="rb") as file:
                self.words_idf_dict = json.loads(file.read().decode("utf-8"))

            logs_idf_dict_path = os.path.join(
                self.model_path, "document_frequency", "logs_document_frequency.gz"
            )
            self.logger.info("Loading logs idf dictionary from: %s", logs_idf_dict_path)
            with gzip.open(logs_idf_dict_path, mode="rb") as file:
                self.logs_idf_dict = json.loads(file.read().decode("utf-8"))

            documents_count_path = os.path.join(
                self.model_path, "document_frequency", "documents_count.txt"
            )
            self.logger.info("Loading documents counter from: %s", documents_count_path)
            with open(documents_count_path, mode="r", encoding="utf-8") as file:
                self.documents_count = int(file.read())

    def __setup_logger(self, log_file: str = None) -> None:
        self.logger = logging.getLogger("LogVectorizer")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] <%(levelname)s> - %(message)s"
        )

        ch = logging.StreamHandler(sys.stdout)  # console
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if not log_file is None:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    @staticmethod
    def split_log_pattern(log: str, dataset_type: str) -> list:
        """Splitting log by tag function"""
        if dataset_type == "HDFS":
            tag = re.findall(REGEX_TAGS_HDFS, log)[0]
        return log.split(tag)[1].split()

    @staticmethod
    def calc_groups_by_interval(logs_group_df):
        """Counter functions in groups"""
        # Расчёт bag of logs
        patterns_group_df = logs_group_df["log_pattern"].value_counts()
        patterns_group_dict = {
            key: int(value) for key, value in dict(patterns_group_df).items()
        }
        # Расчёт bag of words и idf_dict
        counter_words = Counter()
        logs_group_df["log_pattern"].apply(
            partial(LogVectorizer.split_log_pattern, dataset_type="HDFS")
        ).apply(counter_words.update)
        return (
            patterns_group_dict,
            dict(counter_words),
            list(patterns_group_df.index),
            list(counter_words.keys()),
        )

    def __calc_bol_corpus(
        self, logs_df: pd.DataFrame, inference_mode: bool = False
    ) -> Tuple[list, pd.DataFrame]:
        timestamp = pd.DatetimeIndex(logs_df.index, dtype="datetime64[s]").floor(
            f"{self.time_interval}min"
        )
        split_ts = timestamp.unique()
        logs_split_df = pd.DataFrame(index=split_ts[:-1], columns=["logs_count"])

        logs_df["interval"] = pd.cut(
            logs_df.index, bins=split_ts, include_lowest=True, right=False
        )
        logs_df = logs_df.dropna(subset=["interval"])

        logs_split_df["logs_count"] = (
            logs_df.groupby("interval", observed=False)["tag"].count().values
        )

        grouped_logs_df = (
            logs_df.groupby(["interval", "tag"], observed=False)["log_pattern"]
            .count()
            .unstack(level=1)
        )
        grouped_logs_df.index = split_ts[:-1]
        grouped_logs_df.columns.name = None

        logs_split_df = pd.concat([logs_split_df, grouped_logs_df], axis=1)

        bol_corpus, bow_corpus, cur_logs_idf_dict, cur_words_idf_dict = zip(
            *logs_df.groupby("interval", observed=False).apply(
                LogVectorizer.calc_groups_by_interval
            )
        )
        bol_corpus = dict(zip(split_ts[:-1].astype("str"), bol_corpus))
        bow_corpus = dict(zip(split_ts[:-1].astype("str"), bow_corpus))

        if not inference_mode:
            self.documents_count += len(bow_corpus)

            cnt = Counter(self.words_idf_dict)
            for log_int in cur_words_idf_dict:
                if len(log_int) != 0:
                    cnt.update(log_int)
            self.words_idf_dict = dict(cnt)

            cnt = Counter(self.logs_idf_dict)
            for log_int in cur_logs_idf_dict:
                if len(log_int) != 0:
                    cnt.update(log_int)
            self.logs_idf_dict = dict(cnt)
        return bol_corpus, bow_corpus, logs_split_df

    def fit(self, logs_df: pd.DataFrame) -> Tuple[list, pd.DataFrame]:
        """Fitting NLP model function"""
        if self.epochs != 0:
            self.logger.info("Start logs splitting")
            start_time = time.time()
            data_splited = list(
                logs_df["log_pattern"]
                .apply(partial(LogVectorizer.split_log_pattern, dataset_type="HDFS"))
                .values
            )
            self.logger.info(
                "Finished logs splitting, time: %s sec",
                np.round(time.time() - start_time, 2),
            )

        self.logger.info("Start NLP model (%s) fitting", self.nlp_model_name)
        start_time = time.time()
        if self.download_model | (self.path_to_pretrained_nlp_model is not None):
            if self.epochs != 0:
                self.nlp_model.build_vocab(data_splited, update=True)
                self.nlp_model.train(
                    data_splited, total_examples=len(data_splited), epochs=self.epochs
                )
        else:
            if self.epochs == 0:
                self.logger.error(
                    "The number of epochs must be greater than 0 "
                    "when training a model from scratch"
                )
                raise ValueError(
                    "The number of epochs must be greater than 0 "
                    "when training a model from scratch"
                )
            if self.nlp_model_name == "word2vec":
                self.nlp_model = Word2Vec(
                    data_splited,
                    vector_size=self.vector_size,
                    window=self.window,
                    min_count=self.min_count,
                    epochs=self.epochs,
                    workers=self.n_workers,
                )
            elif self.nlp_model_name == "fasttext":
                self.nlp_model = FastText(
                    data_splited,
                    vector_size=self.vector_size,
                    window=self.window,
                    min_count=self.min_count,
                    epochs=self.epochs,
                    workers=self.n_workers,
                )
        self.logger.info(
            "Finished NLP model (%s) fitting, time: %s sec",
            self.nlp_model_name,
            np.round(time.time() - start_time, 2),
        )

        self.logger.info("Start bags of words and logs corpus calculation")
        start_time = time.time()
        bol_corpus, bow_corpus, logs_split_df = self.__calc_bol_corpus(logs_df)
        self.logger.info(
            "Finished bags of word and logs corpus calculation, time: %s sec",
            np.round(time.time() - start_time, 2),
        )
        return bol_corpus, bow_corpus, logs_split_df

    def __load_log_corpus(
        self, corpus_type: str = "bag_of_words", path_to_results: Optional[str] = None
    ) -> dict:
        if corpus_type == "bag_of_words":
            file_startswith = "bow_corpus"
        elif corpus_type == "bag_of_logs":
            file_startswith = "bol_corpus"

        if path_to_results is None:
            files_list = os.listdir(
                os.path.abspath(os.path.join(self.model_path, "train_info"))
            )
        else:
            files_list = os.listdir(
                os.path.abspath(os.path.join(path_to_results, "test_info"))
            )
        files = sorted(
            [file for file in files_list if file.startswith(file_startswith)]
        )

        log_corpus = {}
        for file in files:
            if path_to_results is None:
                log_corpus_path = os.path.abspath(
                    os.path.join(self.model_path, "train_info", file)
                )
            else:
                log_corpus_path = os.path.abspath(
                    os.path.join(path_to_results, "test_info", file)
                )
            with gzip.open(log_corpus_path, mode="rb") as file:
                log_corpus_file = json.loads(file.read().decode("utf-8"))
            log_corpus.update(log_corpus_file)
        if (self.documents_count != len(log_corpus)) & (path_to_results is None):
            self.logger.error(
                "Length of %s corpus (=%s) is not equal to documents_count (=%s)",
                corpus_type,
                len(log_corpus),
                self.documents_count,
            )
            raise ValueError(
                f"Length of {corpus_type} corpus (={len(log_corpus)}) is not equal to "
                f"documents_count (={self.documents_count})"
            )
        return log_corpus

    def __calc_tfidf_dict(
        self, corpus: dict, n_docs: int, corpus_type: str, inference_mode: bool = False
    ) -> None:
        for ts, doc in corpus.items():
            n_doc = sum(doc.values())
            tfidf_doc = {}
            for item, count in doc.items():
                tf = count / n_doc
                if inference_mode:
                    if corpus_type == "bag_of_logs":
                        if item in self.logs_idf_dict:
                            idf = np.log2((n_docs + 2) / (self.logs_idf_dict[item] + 1))
                        else:
                            idf = np.log2(n_docs + 2)
                    else:
                        if item in self.words_idf_dict:
                            idf = np.log2(
                                (n_docs + 2) / (self.words_idf_dict[item] + 1)
                            )
                        else:
                            idf = np.log2(n_docs + 2)
                else:
                    if corpus_type == "bag_of_logs":
                        idf = np.log2((n_docs + 1) / self.logs_idf_dict[item])
                    else:
                        idf = np.log2((n_docs + 1) / self.words_idf_dict[item])
                tfidf_doc[item] = tf * idf
            w_norm = np.sqrt((np.array(list(tfidf_doc.values())) ** 2).sum())
            for item in tfidf_doc:
                tfidf_doc[item] = tfidf_doc[item] / w_norm
            if corpus_type == "bag_of_logs":
                self.logs_tfidf_dict[ts] = tfidf_doc
            elif corpus_type == "bag_of_words":
                self.words_tfidf_dict[ts] = tfidf_doc

    def log_weight_calc(self, log: str, tag: str) -> float:
        """Logs weights calculation by log patterns and tags"""
        weight = 1.0
        if self.patterns_weights is not None:
            if log in self.patterns_weights:
                weight *= self.patterns_weights[log]
        if self.tags_weights is not None:
            if tag in self.tags_weights:
                weight *= self.tags_weights[tag]
            else:
                self.logger.warning(
                    "A new tag was met: %s. It was given a weight: %s",
                    tag,
                    self.tags_weights["###UNKNOWN_TAG"],
                )
                self.tags_weights[tag] = self.tags_weights["###UNKNOWN_TAG"]
                weight *= self.tags_weights["###UNKNOWN_TAG"]
        return weight

    @staticmethod
    def find_tag(logtext: str, dataset_type: str) -> str:
        if dataset_type == "HDFS":
            tag = re.findall(REGEX_TAGS_HDFS, logtext)[0]
        return tag

    def __doc_to_vec(
        self, bol_corpus: dict, calc_logs_importance: bool = False
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        logs_importance_df = []
        doc_vectors = np.zeros((len(bol_corpus), self.vector_size))
        i = 0
        for ts, doc in tqdm(bol_corpus.items()):
            doc_vector = np.zeros(self.vector_size)
            doc_num_logs = 0
            if calc_logs_importance:
                log_vectors = []
                tag_list = []
            for log, n_logs in doc.items():
                tag = LogVectorizer.find_tag(log, dataset_type="HDFS")
                log_vector = np.zeros(self.vector_size)
                log_num_words = 0
                for word in log.split(tag)[1].split():
                    if (
                        (self.nlp_model_name == "word2vec")
                        & (word in self.nlp_model.wv)
                    ) | (self.nlp_model_name == "fasttext"):
                        if self.use_words_tfidf:
                            log_vector += (
                                self.nlp_model.wv[word]
                                * self.words_tfidf_dict[ts][word]
                            )
                        else:
                            log_vector += self.nlp_model.wv[word]
                        log_num_words += 1
                if self.use_logs_tfidf:
                    log_vector *= self.logs_tfidf_dict[ts][log]
                if self.use_logs_patterns:
                    log_vector *= self.log_weight_calc(log, tag)
                if log_num_words != 0:
                    log_vector /= log_num_words
                doc_vector += log_vector * n_logs
                doc_num_logs += n_logs
                if calc_logs_importance:
                    log_vectors.append(log_vector * n_logs)
                    tag_list.append(tag)
            if doc_num_logs != 0:
                doc_vector /= doc_num_logs
                if calc_logs_importance:
                    log_vectors = np.array(log_vectors) / doc_num_logs
                    logs_importance = (doc_vector @ log_vectors.T) / np.linalg.norm(
                        doc_vector
                    ) ** 2
                    logs_importance_df.extend(
                        zip(
                            [ts] * len(doc),
                            list(doc.keys()),
                            tag_list,
                            list(doc.values()),
                            logs_importance,
                        )
                    )
            else:
                doc_vector = np.full(self.vector_size, np.nan)
            doc_vectors[i] = doc_vector
            i += 1
        logs_importance_df = pd.DataFrame(
            logs_importance_df,
            columns=["timestamp", "log_pattern", "tag", "quantity", "log_importance"],
        )
        if calc_logs_importance:
            logs_importance_df["timestamp"] = pd.to_datetime(
                logs_importance_df["timestamp"]
            )
            logs_importance_df["% of all"] = (
                logs_importance_df.groupby("timestamp")
                .apply(lambda x: x["quantity"] / x["quantity"].sum() * 100)
                .reset_index()["quantity"]
            )
            logs_importance_df = logs_importance_df.sort_values(
                ["timestamp", "log_importance"], ascending=[True, False]
            ).reset_index(drop=True)
        return doc_vectors, logs_importance_df

    def __parallel_doc_to_vec(
        self, log_corpus: dict, calc_logs_importance: bool = False
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        random_indexes = np.random.RandomState(42).choice(
            np.arange(len(log_corpus)), len(log_corpus), replace=False
        )
        splitted_keys = np.array_split(
            np.array(list(log_corpus.keys()))[random_indexes], self.n_workers
        )

        splitted_dict = []
        for i in range(self.n_workers):
            log_corpus_i = {}
            for key in splitted_keys[i]:
                log_corpus_i[key] = log_corpus[key]
            splitted_dict.append(log_corpus_i)
        with Pool(self.n_workers) as pool:
            doc_vectors, logs_importance_df = zip(
                *pool.map(
                    partial(
                        self.__doc_to_vec, calc_logs_importance=calc_logs_importance
                    ),
                    splitted_dict,
                )
            )

        sorted_indexes = np.argsort(random_indexes)
        doc_vectors = np.concatenate(doc_vectors)[sorted_indexes]
        logs_importance_df = (
            pd.concat(logs_importance_df)
            .sort_values(["timestamp", "log_importance"], ascending=[True, False])
            .reset_index(drop=True)
        )
        logs_importance_df["% of all"] = (
            logs_importance_df.groupby("timestamp")
            .apply(lambda x: x["quantity"] / x["quantity"].sum() * 100)
            .reset_index()["quantity"]
        )
        return doc_vectors, logs_importance_df

    def __load_logs_split_df(self, path_to_results: str = None) -> pd.DataFrame:
        if path_to_results is None:
            info_folder_path = os.path.join(self.model_path, "train_info")
        else:
            info_folder_path = os.path.join(path_to_results, "test_info")

        files = sorted(
            [
                file
                for file in os.listdir(info_folder_path)
                if file.startswith("logs_split_df")
            ]
        )
        logs_split_df = pd.DataFrame()
        for file in files:
            logs_split_df_path = os.path.join(info_folder_path, file)
            logs_split_df_i = pd.read_csv(
                logs_split_df_path, encoding="utf-8", index_col=0
            )
            if logs_split_df is None:
                logs_split_df = logs_split_df_i
            else:
                logs_split_df = pd.concat([logs_split_df, logs_split_df_i])
        logs_split_df.fillna(0, inplace=True)
        logs_split_df = logs_split_df.astype(int)
        logs_split_df.index = pd.to_datetime(logs_split_df.index)
        return logs_split_df

    def calc_train_vectors(self, parallel_mode: Optional[bool] = False) -> pd.DataFrame:
        """TfIdf dictionary saving and train vectors calculation function"""
        if self.use_words_tfidf:
            self.logger.info(
                "Loading train bag of words corpus from folder: %s",
                os.path.join(self.model_path, "train_info"),
            )
            bow_corpus = self.__load_log_corpus(corpus_type="bag_of_words")

            self.logger.info("Start words TfIdf dictionary calculation")
            start_time = time.time()
            self.__calc_tfidf_dict(
                bow_corpus, n_docs=len(bow_corpus), corpus_type="bag_of_words"
            )
            self.logger.info(
                "Finished words TfIdf dictionary calculation, time: %s sec",
                np.round(time.time() - start_time, 2),
            )

            gc.collect()
            del bow_corpus

        self.logger.info(
            "Loading train bag of logs corpus from folder: %s",
            os.path.join(self.model_path, "train_info"),
        )
        bol_corpus = self.__load_log_corpus(corpus_type="bag_of_logs")

        if self.use_logs_tfidf:
            self.logger.info("Start logs TfIdf dictionary calculation")
            start_time = time.time()
            self.__calc_tfidf_dict(
                bol_corpus, n_docs=len(bol_corpus), corpus_type="bag_of_logs"
            )
            self.logger.info(
                "Finished logs TfIdf dictionary calculation, time: %s sec",
                np.round(time.time() - start_time, 2),
            )

        self.logger.info("Start %s calculation", self.nlp_model_name)
        start_time = time.time()
        if parallel_mode:
            self.logger.info(
                "Using %s workers for parallel calculations", self.n_workers
            )
            doc_vectors, _ = self.__parallel_doc_to_vec(bol_corpus)
        else:
            doc_vectors, _ = self.__doc_to_vec(bol_corpus)
        self.logger.info(
            "Finished %s calculation, time: %s sec",
            self.nlp_model_name,
            np.round(time.time() - start_time, 2),
        )

        self.logger.info(
            "Loading logs counter from folder: %s",
            os.path.join(self.model_path, "train_info"),
        )
        logs_split_df = self.__load_logs_split_df()

        vectors_df = pd.DataFrame(
            doc_vectors,
            index=logs_split_df.index,
            columns=[f"vector_{i + 1}" for i in range(self.vector_size)],
        )
        vectors_df = pd.merge(
            vectors_df, logs_split_df, left_index=True, right_index=True
        )
        return vectors_df

    def calc_test_vectors(
        self,
        logs_df: Optional[pd.DataFrame] = None,
        path_to_results: Optional[str] = None,
        calc_logs_importance: Optional[bool] = False,
        parallel_mode: Optional[bool] = False,
    ) -> pd.DataFrame:
        """Test vectors calculation function"""

        if path_to_results is None:
            self.logger.info("Start bags of words and logs corpus calculation")
            start_time = time.time()
            bol_corpus, bow_corpus, logs_split_df = self.__calc_bol_corpus(
                logs_df, inference_mode=True
            )
            self.logger.info(
                "Finished bags of word and logs corpus calculation, time: %s sec",
                np.round(time.time() - start_time, 2),
            )
        else:
            self.logger.info(
                "Loading logs counter from folder: %s",
                os.path.join(path_to_results, "test_info"),
            )
            logs_split_df = self.__load_logs_split_df(path_to_results)

            self.logger.info(
                "Loading test bag of words corpus from: %s",
                os.path.join(self.model_path, "test_info"),
            )
            bow_corpus = self.__load_log_corpus(
                corpus_type="bag_of_words", path_to_results=path_to_results
            )

            self.logger.info(
                "Loading test bag of logs corpus from: %s",
                os.path.join(self.model_path, "test_info"),
            )
            bol_corpus = self.__load_log_corpus(
                corpus_type="bag_of_logs", path_to_results=path_to_results
            )

        self.logger.info("Start words TfIdf dictionary calculation")
        self.__calc_tfidf_dict(
            bow_corpus,
            n_docs=self.documents_count,
            corpus_type="bag_of_words",
            inference_mode=True,
        )

        self.logger.info("Start logs TfIdf dictionary calculation")
        self.__calc_tfidf_dict(
            bol_corpus,
            n_docs=self.documents_count,
            corpus_type="bag_of_logs",
            inference_mode=True,
        )

        self.logger.info("Start %s calculation", self.nlp_model_name)
        start_time = time.time()
        if parallel_mode:
            self.logger.info(
                "Using %s workers for parallel calculations", self.n_workers
            )
            doc_vectors, logs_importance_corpus = self.__parallel_doc_to_vec(
                bol_corpus, calc_logs_importance=calc_logs_importance
            )
        else:
            doc_vectors, logs_importance_corpus = self.__doc_to_vec(
                bol_corpus, calc_logs_importance=calc_logs_importance
            )
        self.logger.info(
            "Finished %s calculation, time: %s sec",
            self.nlp_model_name,
            np.round(time.time() - start_time, 2),
        )

        test_vectors_df = pd.DataFrame(
            doc_vectors,
            index=logs_split_df.index,
            columns=[f"vector_{i + 1}" for i in range(self.vector_size)],
        )
        test_vectors_df = pd.merge(
            test_vectors_df, logs_split_df, left_index=True, right_index=True
        )
        return (
            test_vectors_df,
            bol_corpus,
            bow_corpus,
            logs_split_df,
            logs_importance_corpus,
        )
