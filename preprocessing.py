import pandas as pd
import os
import re
import nltk


def create_df(dataset, columnnames):
    df = dataset[columnnames]
    df = df.rename(
        columns={columnnames[0]: "character", columnnames[1]: "line"})

    return df


def pool_other(df, n=7):
    names = df.character.value_counts().index.tolist()[:n]
    df["character"] = df["character"].apply(lambda x: "other" if x not in names else x)

    return df

