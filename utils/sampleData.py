import pandas as pd
from functools import partial


def sampleData(df: pd.DataFrame, num_sample: int, groupColumn: str) -> pd.DataFrame:
    def sample(num_sample: int, df: pd.DataFrame):
        return df.sample(num_sample)

    data = df.groupby(groupColumn, as_index=False).apply(partial(sample, num_sample))

    data = data.reset_index(drop=True)

    return data