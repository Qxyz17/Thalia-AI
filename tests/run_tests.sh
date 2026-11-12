#!/usr/bin/env bash
# Thalia test suite - scripted input + grep for expected output.
# Each test uses its own store directory (hermetic).
set -e
cd "$(dirname "$0")/.."
BIN=./build/thalia
[ -f "$BIN" ] || BIN=./build/thalia.exe
PASS=0; FAIL=0
# Scratch files live in the repo: /tmp resolves to different places for
# native binaries and for the shell on Windows.
mkdir -p ./tmpd

check() {  # check <label> <expected_substring>
    if echo "$OUT" | grep -q "$2"; then echo "PASS: $1"; PASS=$((PASS+1));
    else echo "FAIL: $1 (expected '$2' in output)"; FAIL=$((FAIL+1)); fi
}

echo "=== T1: first encounter = no memory ==="
rm -rf t1d
OUT=$(printf '%s\n' 'hello' ':quit' | $BIN --store ./t1d --seed "t1" 2>&1)
check "T1" "no memory"

echo "=== T2: repeat = direct hit ==="
OUT=$(printf '%s\n' 'hello' 'hello' ':quit' | $BIN --store ./t1d --seed "t1" 2>&1)
check "T2" "I remember"

echo "=== T3: persistence across restart ==="
printf '%s\n' 'persistence probe xyz' ':quit' | $BIN --store ./t1d --seed "t1" >/dev/null 2>&1
OUT=$(printf '%s\n' 'persistence probe xyz' ':quit' | $BIN --store ./t1d --seed "t1" 2>&1)
check "T3" "I remember"

echo "=== T4: different inputs -> different keys ==="
rm -rf t4d
KEYS=$(printf '%s\n' 'apple' 'banana' ':quit' | $BIN --store ./t4d --seed "t1" 2>&1 \
       | grep -oE 'key=[a-f0-9]{12}' | sort -u | wc -l)
[ "$KEYS" -ge 2 ] && { echo "PASS: T4"; PASS=$((PASS+1)); } || { echo "FAIL: T4"; FAIL=$((FAIL+1)); }

echo "=== T4b: same-ms writes get unique keys ==="
KEYS=$(printf '%s\n' 'fast' 'fast' 'fast' ':quit' | $BIN --store ./t4d --seed "t1" 2>&1 \
       | grep -oE 'key=[a-f0-9]{12}' | sort -u | wc -l)
[ "$KEYS" -ge 3 ] && { echo "PASS: T4b ($KEYS unique keys)"; PASS=$((PASS+1)); } \
  || { echo "FAIL: T4b ($KEYS unique keys)"; FAIL=$((FAIL+1)); }

echo "=== T5: image recognition (identical bytes = hit) ==="
rm -rf t5d
printf 'unique-aaa\n' > ./tmpd/thalia_t5_a.bin
printf 'unique-bbb\n' > ./tmpd/thalia_t5_b.bin
OUT=$(printf '%s\n' 'img:./tmpd/thalia_t5_a.bin' 'img:./tmpd/thalia_t5_b.bin' 'img:./tmpd/thalia_t5_a.bin' ':quit' \
      | $BIN --store ./t5d --seed "t1" 2>&1)
check "T5a first-see"  "New image"
check "T5b re-see"     "seen this image before"

echo "=== T6: :good raises dopamine ==="
rm -rf t6d
OUT=$(printf '%s\n' 'reward me' ':good' ':state' ':quit' \
      | $BIN --store ./t6d --seed "t1" 2>&1)
D=$(echo "$OUT" | grep 'dopamine=' | sed 's/.*dopamine=//' | cut -c1-4)
awk -v d="$D" 'BEGIN{exit !(d+0 > 0.7)}' && { echo "PASS: T6 (dopamine=$D)"; PASS=$((PASS+1)); } \
  || { echo "FAIL: T6 (dopamine=$D)"; FAIL=$((FAIL+1)); }

echo "=== T7: :sleep consolidates duplicates ==="
rm -rf t7d
OUT=$(printf '%s\n' 'dup phrase' 'dup phrase' ':sleep' ':quit' \
      | $BIN --store ./t7d --seed "t1" 2>&1)
check "T7 merged" "merged=1"

echo "=== T8: sleep persists across restart (snapshot) ==="
printf '%s\n' 'snap marker 42' 'snap marker 42' ':sleep' ':quit' \
  | $BIN --store ./t7d --seed "t1" >/dev/null 2>&1
OUT=$(printf '%s\n' 'snap marker 42' ':quit' | $BIN --store ./t7d --seed "t1" 2>&1)
check "T8" "I remember"

echo "=== T9: silent before learning, speaks after :learn ==="
rm -rf t9d
OUT=$(printf '%s\n' 'tell me something' ':quit' | $BIN --store ./t9d --seed "t1" 2>&1)
check "T9a silent" "no memory"
cat > ./tmpd/thalia_corpus.txt <<'EOF'
hello there friend
hello world said the little robot
the robot dreams about electric sheep every night
good morning world said the robot
EOF
OUT=$(printf ':learn ./tmpd/thalia_corpus.txt\nsay something robot\n:quit\n' | $BIN --store ./t9d --seed "t1" 2>&1)
check "T9b learned" "learned 4 lines"
check "T9c speaks" "voice=yes"
# the speech must not be a fallback template
if echo "$OUT" | grep -q "no memory"; then echo "FAIL: T9d (used template)"; FAIL=$((FAIL+1));
else echo "PASS: T9d"; PASS=$((PASS+1)); fi

echo "=== T10: voice survives :sleep ==="
rm -rf t10d
OUT=$(printf ':learn ./tmpd/thalia_corpus.txt\n:sleep\ntell me something\n:quit\n' | $BIN --store ./t10d --seed "t1" 2>&1)
check "T10 speaks after sleep" "voice=yes"

echo "=== T11: learned corpus persists across restart ==="
OUT=$(printf '%s\n' 'what do you say little robot' ':quit' | $BIN --store ./t9d --seed "t1" 2>&1)
check "T11 voice persists" "voice=yes"

echo "=== T12: semantic similarity via co-occurrence ==="
rm -rf t12d
cat > ./tmpd/thalia_sem.txt <<'EOF'
小狗 在 草地 上 跑
小犬 在 草地 上 跑
狗 喜欢 追 球
犬 喜欢 追 球
石头 很 硬
石头 在 山 上
EOF
OUT=$(printf ':learn ./tmpd/thalia_sem.txt\n:words 狗 犬\n:words 狗 石\n:quit\n' | $BIN --store ./t12d --seed "t12" 2>&1)
check "T12 learned" "learned 6 lines"
S1=$(echo "$OUT" | grep 'sim(狗,犬)' | sed 's/.*= //')
S2=$(echo "$OUT" | grep 'sim(狗,石)' | sed 's/.*= //')
awk -v a="$S1" -v b="$S2" 'BEGIN{exit !(a+0 > b+0 + 0.05)}' \
  && { echo "PASS: T12 synonym beats unrelated (狗犬=$S1 狗石=$S2)"; PASS=$((PASS+1)); } \
  || { echo "FAIL: T12 (狗犬=$S1 狗石=$S2)"; FAIL=$((FAIL+1)); }

echo "=== T13: language statistics persist across restart ==="
OUT=$(printf '%s\n' ':words 狗 犬' ':quit' | $BIN --store ./t12d --seed "t12" 2>&1)
S3=$(echo "$OUT" | grep 'sim(狗,犬)' | sed 's/.*= //')
awk -v a="$S1" -v b="$S3" 'BEGIN{exit !(b+0 > 0.6 && (a-b < 0 ? b-a : a-b) < 0.05)}' \
  && { echo "PASS: T13 sim stable across restart ($S3)"; PASS=$((PASS+1)); } \
  || { echo "FAIL: T13 (before=$S1 after=$S3)"; FAIL=$((FAIL+1)); }

echo "=== T14: sleep re-encodes memories semantically (reconsolidation) ==="
rm -rf t14d
# A memory recorded BEFORE the vocabulary existed (semantically blind),
# then the corpus is learned, then :sleep re-encodes it. A synonym-variant
# query must retrieve it through the ASSOCIATION route (exact stays ~0.5:
# no false "I remember", only "this reminds me").
printf '小狗 每天 都 在 草地 上 玩\n:quit\n' | $BIN --store ./t14d --seed "t14" >/dev/null 2>&1
printf ':learn ./tmpd/thalia_sem.txt\n:sleep\n:quit\n' | $BIN --store ./t14d --seed "t14" >/dev/null 2>&1
OUT=$(printf '小犬 每天 都 在 草地 上 玩\n:quit\n' | $BIN --store ./t14d --seed "t14" 2>&1)
if echo "$OUT" | grep -q 'assoc=0\.[7-9]' && echo "$OUT" | grep -q 'hits=[1-9]'; then
    echo "PASS: T14 synonym recall via association route"; PASS=$((PASS+1))
else
    echo "FAIL: T14"; echo "$OUT" | grep 'hits='; FAIL=$((FAIL+1))
fi

echo "=== T15: no --seed derives one and prints it ==="
rm -rf t15d ./tmpd/thalia_dog.txt
cat > ./tmpd/thalia_dog.txt <<'EOF'
狗 会跑 会叫 喜欢骨头 是 动物
犬 会跑 会叫 喜欢骨头 是 动物
狗 会跑 会叫 忠诚
犬 会跑 会叫 忠诚
猫 喜欢 鱼
猫 喵喵叫
EOF
OUT=$(printf ':learn ./tmpd/thalia_dog.txt\n:sim 狗 犬\n:sim 狗 猫\nhello\n:quit\n' \
      | $BIN --store ./t15d 2>&1)
check "T15a seed printed" "auto seed: Thalia:auto:"
S_DOG=$(echo "$OUT" | grep 'sim(狗,犬)' | sed 's/.*= *//')
S_CAT=$(echo "$OUT" | grep 'sim(狗,猫)' | sed 's/.*= *//')
awk -v a="$S_DOG" 'BEGIN{exit !(a+0 > 0.6)}' \
  && { echo "PASS: T15b sim(狗,犬)=$S_DOG"; PASS=$((PASS+1)); } \
  || { echo "FAIL: T15b (sim(狗,犬)=$S_DOG)"; FAIL=$((FAIL+1)); }
awk -v a="$S_DOG" -v b="$S_CAT" 'BEGIN{exit !(b+0 < a+0)}' \
  && { echo "PASS: T15c dog/cat=$S_CAT < dog/dog=$S_DOG"; PASS=$((PASS+1)); } \
  || { echo "FAIL: T15c (狗犬=$S_DOG 狗猫=$S_CAT)"; FAIL=$((FAIL+1)); }
if echo "$OUT" | grep -q " sim="; then echo "PASS: T15d reply"; PASS=$((PASS+1));
else echo "FAIL: T15d (no reply)"; FAIL=$((FAIL+1)); fi

echo "=== T16: explicit --seed prints no auto seed line ==="
OUT=$(printf '%s\n' 'hello' ':quit' | $BIN --store ./t15d --seed "fixed" 2>&1)
if echo "$OUT" | grep -q "auto seed:"; then echo "FAIL: T16"; FAIL=$((FAIL+1));
else echo "PASS: T16"; PASS=$((PASS+1)); fi

echo "=== T17: 推断未直接说过的话 ==="
rm -rf t17d
cat > ./tmpd/thalia_infer.txt <<'EOF'
动物 会 呼吸 。
狗 是 动物 。
狗 会 跑 。
孩子 是 人 。
人 是 动物 。
人 会 走路 。
EOF
OUT=$(printf ':learn ./tmpd/thalia_infer.txt\n:facts 狗\n:facts 孩子\n:why 孩子\n:quit\n' \
      | $BIN --store ./t17d --seed "t17" 2>&1)
check "T17a learned" "learned 6 lines"
# 狗 是 动物, 动物 会 呼吸 -> 狗 会 呼吸（语料里没这句）
if echo "$OUT" | grep -q '\[推断\] 狗会呼吸'; then echo "PASS: T17b 狗会呼吸"; PASS=$((PASS+1));
else echo "FAIL: T17b (推不出 狗会呼吸)"; FAIL=$((FAIL+1)); fi
# 孩子 是 人, 人 是 动物 -> 孩子 是 动物（两跳）
if echo "$OUT" | grep -q '\[推断\] 孩子是动物'; then echo "PASS: T17c 孩子是动物"; PASS=$((PASS+1));
else echo "FAIL: T17c (推不出 孩子是动物)"; FAIL=$((FAIL+1)); fi
# 推理路径必须给出链条
if echo "$OUT" | grep -q '因为 人是动物'; then echo "PASS: T17d 推理链"; PASS=$((PASS+1));
else echo "FAIL: T17d (没有推理链)"; FAIL=$((FAIL+1)); fi

echo "=== T18: 直接提问走推理而不是复读 ==="
OUT=$(printf '狗 会 什么\n:quit\n' | $BIN --store ./t17d --seed "t17" 2>&1)
if echo "$OUT" | grep -q 'infer=derived'; then echo "PASS: T18 推理应答"; PASS=$((PASS+1));
else echo "FAIL: T18 (没有走推理路径)"; FAIL=$((FAIL+1)); fi

rm -rf t1d t4d t5d t6d t7d t9d t10d t12d t14d t15d t17d ./tmpd/thalia_t5_a.bin ./tmpd/thalia_t5_b.bin ./tmpd/thalia_corpus.txt ./tmpd/thalia_sem.txt ./tmpd/thalia_dog.txt ./tmpd/thalia_infer.txt
echo ""
echo "==============================="
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ] && echo "ALL GREEN" || exit 1
