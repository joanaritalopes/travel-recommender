import pandas as pd

from travel_recsys.config import REPORTS
from travel_recsys.features.f050tags import ACTIVITY_LABELS, model_short_name, output_path

TOP_PAIRS = 10


def label_correlations(scores) -> pd.DataFrame:
    """Spearman rank correlation between every pair of activity labels."""
    return scores[ACTIVITY_LABELS].corr(method="spearman")


def strongest_pairs(corr, n=TOP_PAIRS) -> pd.DataFrame:
    """The n label pairs with the highest correlation, each pair listed once."""
    labels = list(corr.columns)
    pairs = [
        (first, second, float(corr.loc[first, second]))
        for i, first in enumerate(labels)
        for second in labels[i + 1:]
    ]
    table = pd.DataFrame(pairs, columns=["first", "second", "spearman"])
    return table.nlargest(n, "spearman").round(2).reset_index(drop=True)


def main():
    scores = pd.read_csv(output_path(), keep_default_na=False)
    corr = label_correlations(scores)

    REPORTS.mkdir(exist_ok=True)
    out_path = REPORTS / f"label_correlations_{model_short_name()}.csv"
    corr.round(2).to_csv(out_path, index_label="label")

    print(strongest_pairs(corr).to_string(index=False))
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
