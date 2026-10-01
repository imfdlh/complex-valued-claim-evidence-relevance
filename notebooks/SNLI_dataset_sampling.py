import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    # LOAD DATASET
    import pandas as pd
    df = pd.read_parquet(
        "/Users/fadilahnurimani/Documents/Study/ut/smt 5/artikel ilmiah/code/stanfordnli.parquet"
    )
    print(df.shape)
    print(df['label'].value_counts())

    n_per_class = 3000
    df = df[df["label"]!=-1]
    df = (df.groupby("label")
            .apply(lambda x: x.sample(n=min(n_per_class, len(x)), random_state=42))
            .reset_index())
    print(df.shape)
    print(df['label'].value_counts())

    label_map = {0: +1, 1: 0, 2: -1}
    df["y"] = df["label"].map(label_map)

    df.to_parquet("/Users/fadilahnurimani/Documents/Study/ut/smt 5/artikel ilmiah/code/sample_9000_stanfordnli.parquet")
    return (df,)


@app.cell
def _(df):
    df['premise_len']=df['premise'].apply(len)
    df['hypothesis_len']=df['hypothesis'].apply(len)
    return


@app.cell
def _(df):
    df[df['premise_len']<10]
    return


@app.cell
def _(df):
    df[df['hypothesis_len']<10]
    return


if __name__ == "__main__":
    app.run()
