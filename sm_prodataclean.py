import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from datetime import datetime

# ---------------------------------------------
# Load / Create sample dataset
# ---------------------------------------------
data = {
    "review_id": [1, 2, 3, 4],
    "review_text": [
        "<p>This movie was AMAZING!!</p>",
        "Worst movie ever!!      Horrible pacing.",
        "I&nbsp;didn&#39;t like it much...",
        None
    ],
    "rating": [5, None, 2, "4"],
    "timestamp": ["2024/01/02", "02-01-2024", "Jan 2, 2024", "2024-01-02T10:30:00"]
}

df = pd.DataFrame(data)
print("RAW DATA:\n", df, "\n")

# ------------------------------------------------
# 1. Remove HTML tags using BeautifulSoup
# ------------------------------------------------
def remove_html(text):
    if pd.isnull(text):
        return text
    return BeautifulSoup(text, "html.parser").get_text()

df["review_text"] = df["review_text"].apply(remove_html)

# ------------------------------------------------
# 2. Fix HTML entities (like &#39; or &nbsp;)
# ------------------------------------------------
import html
df["review_text"] = df["review_text"].apply(lambda x: html.unescape(x) if isinstance(x, str) else x)

# ------------------------------------------------
# 3. Convert text to lowercase & remove extra spaces
# ------------------------------------------------
df["review_text"] = (
    df["review_text"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# ------------------------------------------------
# 4. Replace missing ratings with median
# ------------------------------------------------
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["rating"] = df["rating"].fillna(df["rating"].median())

# ------------------------------------------------
# 5. Parse timestamps in mixed formats
# ------------------------------------------------
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# ------------------------------------------------
# 6. Remove rows where review text is empty/null
# ------------------------------------------------
df = df.dropna(subset=["review_text"])

# ------------------------------------------------
# Final Cleaned Data
# ------------------------------------------------
print("CLEANED DATA:\n", df)
