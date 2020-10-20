import pandas as pd

import settings
from funcs.utils import find_project_root

ROOT = find_project_root()
SOURCE_PATH = ROOT / settings.path_to_haiku_source
OUTPUT_FILE_PATH = ROOT / settings.path_to_haiku_corpus


def main() -> None:
    source_df = pd.read_csv(SOURCE_PATH)
    print(source_df.head())
    print(source_df.info())
    df = source_df
    # remove ambivalent syllables
    for i in range(3):
        col = f"{i}_syllables"
        df = df.assign(remove=df[col].apply(lambda cell: "," in cell))
        df = df[~df["remove"]]
    df.drop(columns=["remove"])
    # keep only the most popular: 5-5-2
    df = df.assign(
        type=df["0_syllables"] + df["1_syllables"] + df["2_syllables"]
    )
    print(df["type"].value_counts())
    df = df[df["type"] == "575"]
    print(df.info())
    df[["0", "1", "2"]].to_csv(OUTPUT_FILE_PATH, index=False, header=False)


if __name__ == "__main__":
    main()
