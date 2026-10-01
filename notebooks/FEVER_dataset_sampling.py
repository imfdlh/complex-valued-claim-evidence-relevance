import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    # LOAD DATASET
    import pandas as pd
    df = pd.read_parquet(
        "/Users/fadilahnurimani/Documents/Study/ut/smt 5/artikel ilmiah/code/pietrolesci_nli_fever dev-00000-of-00001.parquet"
    )
    print(df.shape)
    print(df['label'].value_counts())

    import numpy as np
    df['hypothesis'] = df['hypothesis'].replace('', np.nan)
    df = df[~df['hypothesis'].isnull()].reset_index(drop=True).copy()

    n_per_class = 3000
    df = df[df["fever_gold_label"]!=-1]
    df = (df.groupby("fever_gold_label")
            .apply(lambda x: x.sample(n=min(n_per_class, len(x)), random_state=42))
            .reset_index())
    print(df.shape)
    print(df['label'].value_counts())

    label_map = {"SUPPORTS": +1, "NOT ENOUGH INFO": 0, "REFUTES": -1}
    df["y"] = df["fever_gold_label"].map(label_map)

    df.to_parquet("/Users/fadilahnurimani/Documents/Study/ut/smt 5/artikel ilmiah/code/sample_9000_fevernli.parquet")
    return (df,)


@app.cell
def _(df):
    df[~(df['hypothesis']==df['hypothesis'].isnull())].isnull().sum()
    return


@app.cell
def _(df):
    df['premise_len']=df['premise'].apply(len)
    df['hypothesis_len']=df['hypothesis'].apply(len)
    return


@app.cell
def _(df):
    df[df['premise_len']<20]
    return


@app.cell
def _(df):
    df[df['hypothesis_len']<50]
    return


if __name__ == "__main__":
    app.run()
