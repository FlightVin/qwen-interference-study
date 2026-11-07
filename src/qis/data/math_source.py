"""Pinned MATH source (EleutherAI/hendrycks_math): loading, stable IDs, gold answers."""
import hashlib
import os
import re

MATH_REPO = "EleutherAI/hendrycks_math"
MATH_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus"]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def last_boxed(text: str):
    """Content of the last \\boxed{...} or \\fbox{...} with balanced braces, else None."""
    starts = [m.end() for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text)]
    if not starts:
        return None
    i = starts[-1]
    depth, j = 1, i
    while j < len(text) and depth:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
        j += 1
    return text[i:j - 1].strip() if depth == 0 else None


def load_math(root: str):
    """Yield one dict per problem from <root>/<subject>/<split>.parquet, in file order."""
    import pyarrow.parquet as pq
    for subject in SUBJECTS:
        for split in ("train", "test"):
            rows = pq.read_table(os.path.join(root, subject, f"{split}.parquet")).to_pylist()
            for idx, r in enumerate(rows):
                gold = last_boxed(r["solution"])
                yield {
                    "id": f"{subject}/{split}/{idx:05d}",
                    "subject": subject,
                    "orig_split": split,
                    "level": r["level"],
                    "type": r["type"],
                    "problem": r["problem"],
                    "solution": r["solution"],
                    "gold": gold,
                    "gold_sha": sha(gold) if gold is not None else None,
                    "problem_sha": sha(r["problem"]),
                }


def download(root: str):
    """Fetch the pinned parquet files (run on compute; needs huggingface_hub)."""
    from huggingface_hub import hf_hub_download
    for subject in SUBJECTS:
        for split in ("train", "test"):
            dst = os.path.join(root, subject, f"{split}.parquet")
            if os.path.exists(dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            p = hf_hub_download(MATH_REPO, f"{subject}/{split}-00000-of-00001.parquet",
                                repo_type="dataset", revision=MATH_REVISION)
            with open(p, "rb") as src, open(dst + ".tmp", "wb") as out:
                out.write(src.read())
            os.replace(dst + ".tmp", dst)
