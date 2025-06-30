from tqdm import tqdm


class SemanticLogGraphModel:
    __version__ = "0.0.1"

    def __init__(
        self,
        log_graph: dict = {},  # начальный граф
        variance_limit: int = 100,  # максимальное количество разных значений в узле по умолчанию
        max_count_any: int = 5,  # сколько значений ##any подряд вызывают прерывание строки
        max_line_tokens: int = 50,  # максимальная глубина графа в токенах (длина строки)
        train_lines: int = 0,
        position_counts: list = [],
    ):
        self.log_graph = log_graph
        self.variance_limit = variance_limit
        self.max_count_any = max_count_any
        self.max_line_tokens = max_line_tokens
        self.subgraph_size = self.calc_subraph_size(self.log_graph)
        self.train_lines = train_lines
        self.position_counts = position_counts
        self.merges = []

    @staticmethod
    def calc_subraph_size(subgraph):
        i = 0
        for _, val in subgraph.items():
            if isinstance(val, dict):
                i += SemanticLogGraphModel.calc_subraph_size(val) + 1
        return i

    @staticmethod
    def merge_subtrees(A, B):
        C = {}
        for key, val in A.items():
            if key == "##count":
                C["##count"] = val + B.get("##count", 0)
            elif key == "##min":
                C["##min"] = min(val, B.get("##min", val))
            elif key == "##max":
                C["##max"] = max(val, B.get("##max", val))
            elif key == "##tr":
                C["##tr"] = val
            if isinstance(val, dict):
                C[key] = SemanticLogGraphModel.merge_subtrees(A[key], B.get(key, {}))
        for key, val in B.items():
            if key not in A.keys():
                C[key] = B[key]
        return C

    def get_variance(self, level):
        if len(self.position_counts) > level:
            return self.position_counts[level]
        else:
            return self.variance_limit

    def process_tokens(self, tokens, level, td, limit_counter):
        if len(tokens) < level + 1:
            return
        if level > self.max_line_tokens:
            return
        if "##tr" in td:
            return  # Do not process terminated branch
        token = tokens[level]

        if "##any" not in td:
            if token not in td:  # New token (create)
                td[token] = {}
        else:
            if level > 0:
                token = "##any"

        token_count = td[token].get("##count", 0)
        td[token]["##count"] = token_count + 1

        variance = self.get_variance(level)
        if len(td.keys()) >= variance:
            allkeys = list(td.keys())
            ratio = len(allkeys) / (td["##count"] + 1)
            if ratio > 0.3 / (level + 1) or len(td.keys()) >= variance * 20:
                mergedebug = {"allkeys": []}
                merged = {}
                childs = []
                for key in allkeys:
                    val = td[key]
                    if isinstance(val, dict):
                        mergedebug["allkeys"].append((key, val.get("##count", 0)))
                        merged = self.merge_subtrees(merged, val)
                        childs.append(key)
                        del td[key]
                td["##any"] = merged
                token = "##any"
                min_key = min(childs)
                max_key = max(childs)
                td["##min"] = min_key[:100]
                td["##max"] = max_key[:100]

                mergedebug["##count"] = td["##count"]
                mergedebug["ratio"] = ratio
                mergedebug["level"] = level
                mergedebug["path"] = tokens[:level]
                self.merges.append(mergedebug)

        if token == "##any":
            limit_counter += 1
            if limit_counter >= self.max_count_any:
                td["##tr"] = True
                td["##any"] = {}
                return
        else:
            limit_counter = 0

        self.process_tokens(tokens, level + 1, td[token], limit_counter)

    def preprocess_logline_tokens(self, logline: str, any_len: int = 25):
        tokens, sysline = self.process_line(logline, return_sysline=True)
        subtree = self.log_graph
        tokens = tokens[: self.max_line_tokens + 1]
        for i in range(len(tokens)):
            if "##any" in subtree:
                tokens[i] = (
                    f"##any{subtree['##min'][:any_len]}#{subtree['##max'][:any_len]}"
                )
                token = "##any"
            else:
                token = tokens[i]

            if "##tr" in subtree:
                tokens = tokens[: i + 1]
                break
            if token not in subtree:
                break  # unknown token
            subtree = subtree[token]
        return sysline, tokens

    def process_line(self, logline: str, return_sysline: bool = True):
        """
        Разделяет строку на токены.
        Если нужно возвращает системную часть
        ## HDFS SPECIFIC CODE
        """
        tokens = logline.split(" ")[2:]
        sysline = " ".join(logline.split(" ")[:2])

        if return_sysline:
            return tokens, sysline
        else:
            return tokens

    def fit(self, logfile: list, lines_limit=None, count_interval: int = 100000):
        if "##count" not in self.log_graph:
            self.log_graph["##count"] = 0
        i = 0
        for logline in tqdm(logfile):
            tokens = self.process_line(logline, return_sysline=False)
            self.process_tokens(tokens, 0, self.log_graph, 0)
            i += 1

            if (count_interval) and (i % count_interval == 0):
                self.subgraph_size = self.calc_subraph_size(self.log_graph)

            if (lines_limit) and (i >= lines_limit):
                break
        self.log_graph["##count"] += i
        self.train_lines += i
        return

    def preprocess_logline(self, logline: str, any_len: int = 25):
        tokens, sysline = self.process_line(logline, return_sysline=True)
        subtree = self.log_graph
        tokens = tokens[: self.max_line_tokens + 1]
        for i in range(len(tokens)):
            if "##any" in subtree:
                tokens[i] = (
                    f"##any{subtree['##min'][:any_len]}#{subtree['##max'][:any_len]}"
                )
                token = "##any"
            else:
                token = tokens[i]

            if "##tr" in subtree:
                tokens = tokens[: i + 1]
                break
            if token not in subtree:
                break  # unknown token
            subtree = subtree[token]
        return sysline + " " + " ".join(tokens)
