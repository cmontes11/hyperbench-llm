from datasets import load_dataset
from itertools import islice

# Dataset indices range
start = 100_900   
end   = 101_000  

# 1. Load the split with streaming
ds = load_dataset("bigcode/the-stack-smol", split="train", streaming=True)

# 2. Read only indexed lines
slice_iter = islice(ds, start, end)

# 3. Write data to file
with open("eval_slice.txt", "w", encoding="utf-8") as f:
    for row in slice_iter:
        content = row["content"].strip()
        if content:
            f.write(content + "\n")

print("✅ Done! Wrote lines {} to {} to eval_slice.txt".format(start + 1, end))
