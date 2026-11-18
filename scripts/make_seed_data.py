"""Generate a small SYNTHETIC English training set (data/train_seed.csv).

This only exists so the app works out of the box. For real accuracy, train on a real
dataset (Amazon / Flipkart reviews from Kaggle) with:  python train_model.py --data your.csv
"""
import random
from pathlib import Path

import pandas as pd

random.seed(42)

PRODUCTS = ["phone", "laptop", "headphones", "earbuds", "watch", "camera", "speaker",
            "charger", "keyboard", "mouse", "tv", "shoes", "backpack", "blender", "router"]
ASPECTS = ["battery", "sound quality", "build quality", "display", "delivery", "packaging",
           "price", "performance", "design", "customer service", "camera", "fit"]

POS_OPEN = ["Absolutely love this {p}.", "Great {p}!", "Very happy with this {p}.",
            "Excellent {p}, totally worth it.", "Best {p} I have bought so far.",
            "Superb {p} for the price.", "Really impressed by this {p}.",
            "Fantastic {p}, highly recommended.", "Amazing {p}, works perfectly."]
POS_ASPECT = ["The {a} is excellent.", "The {a} is amazing.", "{a} is really good.",
              "I am very satisfied with the {a}.", "The {a} exceeded my expectations.",
              "Loved the {a}.", "{a} is outstanding and great value."]
NEG_OPEN = ["Terrible {p}.", "Very disappointed with this {p}.", "Worst {p} ever.",
            "Waste of money, this {p} is bad.", "Do not buy this {p}.",
            "Poor {p}, stopped working in a week.", "Horrible {p}, I want a refund.",
            "Complete garbage, the {p} broke quickly.", "Not worth it, bad {p}."]
NEG_ASPECT = ["The {a} is terrible.", "The {a} is very poor.", "{a} is awful.",
              "I am unhappy with the {a}.", "The {a} was a big letdown.",
              "Hated the {a}.", "{a} is not good at all."]
NEU_PHRASES = ["It is okay, nothing special.", "Average {p}, does the job.",
               "The {p} is fine for the price, neither great nor bad.",
               "Works as described, nothing more.", "Decent {p} but not amazing.",
               "It is an ordinary {p}.", "Not bad, not great either.",
               "The {a} is average.", "{a} is okay, I expected a bit more.",
               "Acceptable {p}, you get what you pay for."]
MIXED = ["The {a} is great but the {b} is poor.", "Good {a}, however the {b} is disappointing.",
         "I like the {a} but the {b} is bad.", "Excellent {a}, terrible {b}.",
         "The {b} is awful though the {a} is nice."]
POS_ADJ = ["excellent", "amazing", "great", "fantastic", "superb", "wonderful", "perfect",
           "fast", "smooth", "sturdy", "comfortable", "crisp", "reliable", "impressive", "lovely",
           "durable", "solid", "premium", "quick", "nice", "good", "awesome", "brilliant"]
NEG_ADJ = ["terrible", "awful", "horrible", "poor", "bad", "disappointing", "useless", "flimsy",
           "slow", "damaged", "defective", "cheap", "broken", "late", "unreliable", "faulty",
           "worthless", "annoying", "uncomfortable", "overpriced", "rubbish", "pathetic"]
POS_TPL = ["{A} {a}, {adj} {p}.", "The {a} is {adj} and the {b} is {adj2}.", "{adj} {a} and {adj2} {b}.",
           "Got it quickly, the {a} was {adj}.", "{adj} {p}, would buy again.", "Love it, {adj} {a}."]
NEG_TPL = ["{A} was {nadj}, {nadj2} {p}.", "The {a} is {nadj} and the {b} is {nadj2}.", "{nadj} {a}, {nadj2} {b}.",
           "Arrived late and {nadj}, the {a} is {nadj2}.", "{nadj} {p}, would not buy again.", "Hate it, {nadj} {a}."]
NEGATED_POS = ["The {a} is not good.", "Not happy with the {a}.", "{a} is not great at all.",
               "This {p} is not worth the money.", "I would not recommend this {p}."]
NEGATED_NEG = ["The {a} is not bad.", "Not bad at all, the {a} is fine."]


def fill(t):
    a, b = random.sample(ASPECTS, 2)
    adj, adj2 = random.sample(POS_ADJ, 2)
    nadj, nadj2 = random.sample(NEG_ADJ, 2)
    return t.format(p=random.choice(PRODUCTS), a=a.capitalize() if t.startswith("{a}") else a,
                    A=a.capitalize(), b=b, adj=adj, adj2=adj2, nadj=nadj, nadj2=nadj2)


def build(n_each=450):
    rows = []
    for _ in range(n_each):
        pos = fill(random.choice(POS_OPEN)) + " " + fill(random.choice(POS_ASPECT))
        neg = fill(random.choice(NEG_OPEN)) + " " + fill(random.choice(NEG_ASPECT))
        rows += [(pos, "positive"), (neg, "negative")]
        rows += [(fill(random.choice(POS_TPL)), "positive"), (fill(random.choice(NEG_TPL)), "negative")]
    for _ in range(n_each):
        neu = fill(random.choice(NEU_PHRASES))
        if random.random() < 0.5:
            neu = fill(random.choice(MIXED))
        rows.append((neu, "neutral"))
    for _ in range(n_each // 3):
        rows.append((fill(random.choice(NEGATED_POS)), "negative"))
        rows.append((fill(random.choice(NEGATED_NEG)), "neutral"))
    random.shuffle(rows)
    return pd.DataFrame(rows, columns=["text", "label"]).drop_duplicates("text")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "train_seed.csv"
    df = build()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
    print(df["label"].value_counts())
