

import regex as re
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import time


from tests.common import gpt2_bytes_to_unicode


# 编译正则表达式，用于预分词
# 这个模式会匹配常见的缩写、单词、数字、标点符号和空格
# pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
pat = re.compile("""'(?:[sdmt]|ll|ve|re)| ?[\w]+| ?[0-9]+| ?[^\s\w]+|\s+""")
# 初始化字节编码器和解码器
byte_encoder = gpt2_bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}

# ---------- 训练前：保护特殊 token ----------
PLACEHOLDER_PREFIX = "▁"


def _protect(text: str, special_tokens: List[str]) -> Tuple[str, Dict[str, str]]:
    """
    将文本中的特殊 token 替换为唯一的占位符，以防止它们被拆分。
    返回处理后的文本和占位符到原始 token 的映射。
    """
    ph2tok = {}
    # 按长度降序排序，防止短 token 是长 token 的子串时被错误替换
    for i, tok in enumerate(sorted(special_tokens, key=len, reverse=True)):
        ph = f"{PLACEHOLDER_PREFIX}{i}"
        ph2tok[ph] = tok
        text = text.replace(tok, ph)
    return text, ph2tok


# ---------- 分词 ----------
def pre_tokenize(text: str) -> List[str]:
    """
    使用预定义的正则表达式对文本进行初步分词。
    返回一个字符串列表。
    """
    return re.findall(pat, text)


# ---------- 优化的BPE核心逻辑 ----------
def _get_pairs(word: List[bytes]) -> Set[Tuple[bytes, bytes]]:
    """获取单词中的所有相邻字节对"""
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


class OptimizedBPETrainer:
    """优化的BPE训练器，使用增量更新策略来提高效率。"""

    def __init__(self):
        self.pair2score: Dict[Tuple[bytes, bytes], int] = {}
        self.word2splits: Dict[str, List[bytes]] = {}
        self.word2count: Dict[str, int] = {}

    def initialize(self, word2count: Dict[str, int], word2splits: Dict[str, List[bytes]]):
        """初始化训练器的状态。"""
        self.word2count = word2count.copy()
        self.word2splits = {w: s[:] for w, s in word2splits.items()}
        self._compute_all_pair_scores()

    # def _compute_all_pair_scores(self):
    #     """计算所有单词中所有字节对的初始分数（频率）。"""
    #     self.pair2score = {}
    #     for word, count in self.word2count.items():
    #         splits = self.word2splits[word]
    #         for pair in _get_pairs(splits):
    #             self.pair2score[pair] = self.pair2score.get(pair, 0) + count
    
    def _compute_all_pair_scores(self):
        """计算所有单词中所有字节对的初始分数（频率）。"""
        self.pair2score = {}
        for word, count in self.word2count.items():
            splits = self.word2splits[word]
            # 修复：直接遍历所有相邻位置
            for i in range(len(splits) - 1):
                pair = (splits[i], splits[i + 1])
                self.pair2score[pair] = self.pair2score.get(pair, 0) + count

    # def _update_pair_scores_for_word(self, word: str, old_splits: List[bytes], new_splits: List[bytes]):
    #     """
    #     为单个单词更新 pair 分数（增量更新）。
    #     减去旧 pair 的分数，加上新 pair 的分数。
    #     """
    #     count = self.word2count[word]

    #     # 减去旧 pair 的分数
    #     for pair in _get_pairs(old_splits):
    #         self.pair2score[pair] -= count
    #         if self.pair2score[pair] == 0:
    #             del self.pair2score[pair]

    #     # 加上新 pair 的分数
    #     for pair in _get_pairs(new_splits):
    #         self.pair2score[pair] = self.pair2score.get(pair, 0) + count
    def _update_pair_scores_for_word(self, word: str, old_splits: List[bytes], new_splits: List[bytes]):
        """
        为单个单词更新 pair 分数（增量更新）。
        减去旧 pair 的分数，加上新 pair 的分数。
        """
        count = self.word2count[word]

        # 修复：减去旧 pairs（遍历所有位置）
        for i in range(len(old_splits) - 1):
            pair = (old_splits[i], old_splits[i + 1])
            self.pair2score[pair] -= count
            if self.pair2score[pair] == 0:
                del self.pair2score[pair]

        # 修复：加上新 pairs（遍历所有位置）
        for i in range(len(new_splits) - 1):
            pair = (new_splits[i], new_splits[i + 1])
            self.pair2score[pair] = self.pair2score.get(pair, 0) + count
    def _merge_best_pair(self) -> Tuple[bytes, bytes]:
        """
        找到分数最高的 pair，并在所有相关单词中执行合并。
        返回被合并的 pair。
        """
        if not self.pair2score:
            return None

        # 找到最佳 pair
        best_pair = max(self.pair2score, key=self.pair2score.get)
        a, b = best_pair

        # 找出所有包含该 pair 的单词
        affected_words = [
            word for word, splits in self.word2splits.items()
            if best_pair in _get_pairs(splits)
        ]

        # 只更新受影响的单词
        for word in affected_words:
            old_splits = self.word2splits[word]
            new_splits = self._merge_pair_in_word(a, b, old_splits)
            self._update_pair_scores_for_word(word, old_splits, new_splits)
            self.word2splits[word] = new_splits

        return best_pair

    def _merge_pair_in_word(self, a: bytes, b: bytes, splits: List[bytes]) -> List[bytes]:
        """在单个单词的拆分列表中合并指定的 pair。"""
        if len(splits) < 2:
            return splits

        new_word = []
        i = 0
        ab = a + b
        while i < len(splits):
            if i < len(splits) - 1 and splits[i] == a and splits[i + 1] == b:
                new_word.append(ab)
                i += 2
            else:
                new_word.append(splits[i])
                i += 1
        return new_word

    def merge_best_pair(self) -> Tuple[bytes, bytes]:
        """
        找到分数最高的 pair，并在所有相关单词中执行合并。
        返回被合并的 pair。
        """
        if not self.pair2score:
            return None

        # 找到最高分
        max_score = max(self.pair2score.values())

        # 获取所有最高分的 pair
        best_pairs = [pair for pair, score in self.pair2score.items() if score == max_score]

        # 对这些 pair 按字典序排序，确保稳定性
        best_pair = max(best_pairs, key=lambda p: (p[0], p[1]))  # 按第一个字节，再按第二个字节排序

        # print("Merging:", best_pair, "with score", self.pair2score[best_pair])
        a, b = best_pair

        # 找出所有包含该 pair 的单词
        affected_words = [
            word for word, splits in self.word2splits.items()
            if best_pair in _get_pairs(splits)
        ]

        # 只更新受影响的单词
        for word in affected_words:
            old_splits = self.word2splits[word]
            new_splits = self._merge_pair_in_word(a, b, old_splits)
            self._update_pair_scores_for_word(word, old_splits, new_splits)
            self.word2splits[word] = new_splits

        return best_pair

# ---------- 主训练函数 ----------
def train_bpe(corpus_file: str,
              vocab_size: int,
              special_tokens: List[str] = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    训练一个BPE分词器。

    Args:
        corpus_file: 训练语料库的文件路径。
        vocab_size: 目标词汇表大小。
        special_tokens: 需要保护的特殊 token 列表。

    Returns:
        一个元组，包含 (词汇表, 合并操作列表)。
        词汇表是一个字典 {token_id: token_bytes}。
        合并操作列表是一个列表，包含所有合并的 pair。
    """
    if special_tokens is None:
        special_tokens = []

    # 1. 读文件 + 保护特殊 token
    protected_lines = []
    ph2tok = {}
    with open(corpus_file, encoding="utf-8") as f:
        for line in f:
            line, tmp_ph2tok = _protect(line.strip(), special_tokens)
            ph2tok.update(tmp_ph2tok)
            protected_lines.append(line)

    # 2. 正则分词
    tokens = []
    for line in protected_lines:
        tokens.extend(pre_tokenize(line))

    # 3. 字节化并统计词频 - 只保留编码后字节数>=2的token，因为只有它们才能产生字节对
    word2count = defaultdict(int)
    for tok in tokens:
        byte_seq = [c.encode("utf-8") for c in tok]
        if len(byte_seq) >= 2:  # 只处理编码后字节数>=2的token
            word2count[tok] += 1

    # 过滤掉字节数<2的token
    word2splits = {}
    for w in word2count:
        byte_seq = [c.encode("utf-8") for c in w]
        if len(byte_seq) >= 2:
            word2splits[w] = byte_seq

    # 4. 初始 vocab = 所有单字节
    # 初始词汇表大小为 256
    vocab = {i: bytes([b]) for i, b in enumerate(byte_encoder.keys())}
    merges: List[Tuple[bytes, bytes]] = []

    # 检查是否有足够的token来训练BPE
    if not word2splits:
        print("Warning: No tokens with byte length >= 2 found in corpus. BPE training cannot proceed.")
        # 如果没有可训练的token，直接添加特殊token并返回
        for tok in special_tokens:
            vocab[len(vocab)] = tok.encode("utf-8")
        return vocab, merges

    # 5. 使用优化的训练器进行BPE训练
    trainer = OptimizedBPETrainer()
    trainer.initialize(word2count, word2splits)

    # 6. BPE训练循环
    # 循环直到词汇表达到目标大小（减去特殊token的数量）
    target_merges = vocab_size - len(vocab) - len(special_tokens)
    print(f"Starting training. Initial vocab size: {len(vocab)}. Target merges: {target_merges}")
    print(f"Number of tokens to process: {len(word2splits)}")

    for i in range(target_merges+1):
        if not trainer.pair2score:
            print("No more pairs to merge. Stopping early.")
            break

        best_pair = trainer.merge_best_pair()
        if best_pair is None:
            print("Could not find best pair. Stopping early.")
            break

        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token
        merges.append(best_pair)

    # 7. 将特殊 token 添加到词汇表
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")

    print(f"Training finished. Final vocab size: {len(vocab)}. Total merges: {len(merges)}.")
    return vocab, merges