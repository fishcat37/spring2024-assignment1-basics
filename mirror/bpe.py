"""
Optimized BPE Tokenizer Training Implementation
"""
import regex as re
import heapq
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set, Optional
from pathlib import Path

# 使用测试中的相同函数以确保结果一致
from tests.common import gpt2_bytes_to_unicode


def get_gpt2_word_pat(special_tokens: List[str]) -> re.Pattern:
    """GPT2 风格的分词正则表达式"""
    pattern_parts = [re.escape(tok) for tok in special_tokens]
    special_pat = '|'.join(pattern_parts) if pattern_parts else ''

    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    if special_pat:
        return re.compile(special_pat + pat)
    return re.compile(pat)


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: Optional[List[str]] = None
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    优化的 BPE 训练实现
    使用频率优先队列 + 增量更新
    """
    if special_tokens is None:
        special_tokens = []

    # 初始化词汇表
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256

    # 添加特殊 token
    existing_bytes: Set[bytes] = set(vocab.values())
    for tok in special_tokens:
        tok_bytes = tok.encode('utf-8')
        if tok_bytes in existing_bytes:
            raise ValueError(f"Special token {tok} conflicts with existing byte values.")
        vocab[next_id] = tok_bytes
        next_id += 1
        existing_bytes.add(tok_bytes)

    # 读取文本
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        raise IOError(f"Error reading input file: {e}")

    # 使用 GPT2 模式分词
    pat = get_gpt2_word_pat(special_tokens)

    # 将文本转换为字节序列,并统计频率
    token_freq: Dict[Tuple[bytes, ...], int] = defaultdict(int)

    for match in pat.finditer(text):
        word = match.group()
        if not word:
            continue

        if word in special_tokens:
            token_freq[(word.encode('utf-8'),)] += 1
            continue

        word_bytes = word.encode('utf-8')
        byte_seq = tuple(bytes([b]) for b in word_bytes)
        if byte_seq:
            token_freq[byte_seq] += 1

    # 计算初始 pair 计数
    pair_counts: Dict[Tuple[bytes, bytes], int] = defaultdict(int)
    for token_seq, freq in token_freq.items():
        for i in range(len(token_seq) - 1):
            pair = (token_seq[i], token_seq[i + 1])
            pair_counts[pair] += freq

    merges: List[Tuple[bytes, bytes]] = []

    # 主循环
    while len(vocab) < vocab_size and pair_counts:
        # 使用 max 找到最高频的 pair（正确处理平局情况）
        # 按频率降序，频率相同按pair的字节值降序（与参考实现一致）
        max_freq = max(pair_counts.values())
        candidates = [p for p, f in pair_counts.items() if f == max_freq]
        pair = max(candidates)  # 选择最大的（按字节顺序）

        # 执行合并
        freq = pair_counts[pair]
        new_token = pair[0] + pair[1]  # bytes
        vocab[next_id] = new_token
        merges.append((pair[0], pair[1]))
        next_id += 1

        # 增量更新:只更新受影响的 token 序列
        for token_seq, seq_freq in list(token_freq.items()):
            # 检查这个序列是否包含 pair
            found_idx = -1
            for i in range(len(token_seq) - 1):
                if token_seq[i] == pair[0] and token_seq[i + 1] == pair[1]:
                    found_idx = i
                    break

            if found_idx == -1:
                continue

            # 找到包含 pair 的序列,从 pair_counts 中减去这个序列的贡献
            for j in range(len(token_seq) - 1):
                old_pair = (token_seq[j], token_seq[j + 1])
                pair_counts[old_pair] -= seq_freq
                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]

            # 创建新的合并后的序列
            new_seq = []
            i = 0
            while i < len(token_seq):
                if i < len(token_seq) - 1 and \
                   token_seq[i] == pair[0] and \
                   token_seq[i + 1] == pair[1]:
                    new_seq.append(new_token)
                    i += 2
                else:
                    new_seq.append(token_seq[i])
                    i += 1

            new_seq_tuple = tuple(new_seq)
            token_freq[new_seq_tuple] = token_freq.pop(token_seq)

            # 更新新序列的 pair 计数
            for j in range(len(new_seq) - 1):
                new_pair = (new_seq[j], new_seq[j + 1])
                pair_counts[new_pair] += seq_freq

    return vocab, merges
