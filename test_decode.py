#!/usr/bin/env python3
"""
测试 decode 方法

已发现的 decode 方法的 bug（在 cs336_basics/tokenizer.py 第 194-198 行）：

1. 第 195 行：使用 `byte`（整数）作为键查找 `byte_decoder`
   - 但 `byte_decoder` 的键是字符（如 'a'），不是整数（如 97）
   - 应该先将 token bytes 用 UTF-8 解码为字符串，然后用字符查找

2. 第 198 行：当 byte 不在 `byte_decoder` 中时
   - 代码追加的是 `byte`（整数），导致最后 join 失败
   - 应该追加 `bytes([byte])`

正确的 decode 逻辑应该是：
1. 获取 token bytes
2. 用 UTF-8 解码得到 GPT unicode 字符串
3. 遍历字符串中的每个字符，用 byte_decoder 转换回原始字节
4. 拼接所有原始字节并用 UTF-8 解码
"""

import json
import sys
from cs336_basics.tokenizer import Tokenizer

# 设置输出编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8')


def load_tokenizer(vocab_path: str, merges_path: str, special_tokens: list = None):
    """从文件加载 tokenizer"""
    with open(vocab_path, "r", encoding="utf-8") as f:
        readable_vocab = json.load(f)
    vocab = {int(idx): token.encode("utf-8") for idx, token in readable_vocab.items()}

    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            token1, token2 = line.strip().split()
            merges.append((token1.encode("utf-8"), token2.encode("utf-8")))

    return Tokenizer(vocab, merges, special_tokens)


def demonstrate_bug():
    """演示 decode 方法的 bug"""
    tokenizer = load_tokenizer("vocab.json", "merges.txt", special_tokens=["<|endoftext|>"])

    # 查看 byte_decoder 的结构
    print("=" * 60)
    print("分析 byte_decoder 结构")
    print("=" * 60)
    print(f"byte_decoder 类型: {type(list(tokenizer.byte_decoder.items())[0][0])}")
    print(f"byte_decoder 键示例: {list(tokenizer.byte_decoder.items())[0]}")

    # 测试单个字符 'a'
    test_id = 97
    token = tokenizer.id_to_token.get(test_id)
    print(f"\n测试 token ID {test_id}: {token!r}")

    text_bytes = b"".join([token])
    print(f"text_bytes: {list(text_bytes)}")

    # 演示 bug
    print("\n--- 演示 decode 方法的 bug ---")
    for byte in text_bytes:
        print(f"遍历 byte: {byte} (类型: {type(byte).__name__})")
        print(f"  byte in byte_decoder: {byte in tokenizer.byte_decoder}  # 永远是 False!")

        char = chr(byte)
        print(f"  chr(byte) in byte_decoder: {char in tokenizer.byte_decoder}  # 应该是这个!")
        if char in tokenizer.byte_decoder:
            print(f"    byte_decoder[chr(byte)]: {tokenizer.byte_decoder[char]}")


def test_decode_workaround():
    """测试使用 workaround 的 decode"""
    tokenizer = load_tokenizer("vocab.json", "merges.txt", special_tokens=["<|endoftext|>"])

    def fixed_decode(ids):
        """修复后的 decode 实现"""
        token_list = []
        for id in ids:
            token = tokenizer.id_to_token.get(id)
            if token is None:
                unk_id = tokenizer.token_to_id.get(b"<unk>")
                if unk_id is None:
                    raise ValueError(f"Token ID {id} not found in vocab and no <unk> defined.")
                token = tokenizer.id_to_token.get(unk_id)
            token_list.append(token)

        # 关键修复：先拼接 token bytes，然后用 UTF-8 解码为 GPT unicode 字符串
        # 然后遍历每个字符，用 byte_decoder 转换回原始字节
        text_bytes = b"".join(token_list)
        gpt_unicode_str = text_bytes.decode("utf-8")  # GPT unicode 字符串

        decoded_chars = []
        for char in gpt_unicode_str:
            if char in tokenizer.byte_decoder:
                decoded_chars.append(bytes([tokenizer.byte_decoder[char]]))
            else:
                # 如果字符不在 byte_decoder 中，直接用 UTF-8 编码
                decoded_chars.append(char.encode("utf-8"))

        return b"".join(decoded_chars).decode("utf-8")

    print("\n" + "=" * 60)
    print("测试修复后的 decode")
    print("=" * 60)

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
            decoded = fixed_decode(ids)
            match = "[OK]" if text == decoded else "[FAIL]"
            results.append(f"{match} '{text}' -> ids: {ids} -> '{decoded}'")
        except Exception as e:
            results.append(f"[ERROR] '{text}' -> error: {e}")

    return results


if __name__ == "__main__":
    demonstrate_bug()
    results = test_decode_workaround()
    for r in results:
        print(r)

    # 保存结果到文件
    with open("test_decode_results.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("测试修复后的 decode 结果\n")
        f.write("=" * 60 + "\n")
        for r in results:
            f.write(r + "\n")
    print("\n结果已保存到 test_decode_results.txt")
