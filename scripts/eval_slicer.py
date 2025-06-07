from datasets import load_dataset
from itertools import islice

# NOTE: Python is 0-indexed, so line 100,000 is index 99_999
start = 100_900   # 100,000th line (inclusive)
end   = 101_000  # 101,000th line (exclusive)

# 1. Load the split with streaming
ds = load_dataset("bigcode/the-stack-smol", split="train", streaming=True)

# 2. Grab just the lines you want
slice_iter = islice(ds, start, end)

# 3. Write them to file
with open("eval_slice.txt", "w", encoding="utf-8") as f:
    for row in slice_iter:
        content = row["content"].strip()
        if content:
            f.write(content + "\n")

print("✅ Done! Wrote lines {} to {} to eval_slice.txt".format(start + 1, end))
