#!/usr/bin/env python3
"""测试用户新写的 decode 方法"""

import json
import sys
from cs336_basics.tokenizer import Tokenizer

sys.stdout.reconfigure(encoding='utf-8')


def load_tokenizer(vocab_path: str, merges_path: str, special_tokens: list = None):
    with open(vocab_path, "r", encoding="utf-8") as f:
        readable_vocab = json.load(f)
    vocab = {int(idx): token.encode("utf-8") for idx, token in readable_vocab.items()}

    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            token1, token2 = line.strip().split()
            merges.append((token1.encode("utf-8"), token2.encode("utf-8")))

    return Tokenizer(vocab, merges, special_tokens)


def test_decode():
    tokenizer = load_tokenizer("vocab.json", "merges.txt", special_tokens=["<|endoftext|>"])

    test_cases = [
        "",
        "a",
        "the",
        "hello world",
        "Hello, how are you?",
        "The quick brown fox jumps over the lazy dog.",
        "你好",
        "<|endoftext|> hello",
        "hello <|endoftext|> world",
    ]

    results = []
    for text in test_cases:
        try:
            ids = tokenizer.encode(text)
            decoded = tokenizer.decode(ids)  # 直接调用 tokenizer 的 decode
            match = "[OK]" if text == decoded else "[FAIL]"
            results.append(f"{match} '{text}' -> ids: {ids} -> '{decoded}'")
        except Exception as e:
            results.append(f"[ERROR] '{text}' -> error: {e}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("测试用户新写的 decode 方法")
    print("=" * 60)
    results = test_decode()
    for r in results:
        print(r)

    with open("test_decode_new_results.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("测试用户新写的 decode 方法\n")
        f.write("=" * 60 + "\n")
        for r in results:
            f.write(r + "\n")
    print("\n结果已保存到 test_decode_new_results.txt")
