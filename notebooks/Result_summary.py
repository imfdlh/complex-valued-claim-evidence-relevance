import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.function
def compute_features(theta_deg):

    import math
    import cmath
    import numpy as np

    cos_sim = math.cos(math.radians(theta_deg))

    theta_rad = np.arccos(cos_sim)

    r = abs(cos_sim)
    Z = r * np.exp(1j * theta_rad)
    e_i_theta = np.exp(1j * theta_rad)

    k = theta_rad / np.pi
    polar_form = f"{r:.4f} e^(i {k:.4f}π)"

    return {
        "cosine":    f"{cos_sim:.5f}",
        "r":         f"{r:.5f}",
        "theta_deg": f"{theta_deg:.5f}",
        "r, theta": (f"{cmath.polar(Z)[0]:.5f}", f"{math.degrees(cmath.polar(Z)[1]):.5f}"),
        "e^(u*theta)": f"{e_i_theta:.5f}",
        "polar": polar_form,
        "Re":        f"{Z.real:.5f}",
        "Im":        f"{Z.imag:.5f}",
        "Z":         f"{Z:.5f}",
    }


@app.cell
def _():
    import pandas as pd
    df = pd.DataFrame({
        "theta_degree":[0,30,45,60,90,120,135,150,180]
    })
    df[['cosine', 'r', 'theta_deg', 'r, theta', 'e^(u*theta)', 'polar', 'Re',
           'Im', 'Z']]=df["theta_degree"].apply(lambda x: compute_features(x)).apply(pd.Series)
    df
    return (pd,)


@app.cell
def _(pd):
    df_f1 = {}
    df_f1["snli-all-MiniLM-L6-v2"]= pd.read_parquet("/Users/fadilahnurimani/Documents/Study/ut/smt 5/artikel ilmiah/code/snli-all-MiniLM-L6-v2-f1.parquet")
    df_f1["snli-mxbai-embed-large-v1"]= pd.read_parquet("/Users/fadilahnurimani/Documents/Study/ut/smt 5/artikel ilmiah/code/snli-mxbai-embed-large-v1-f1.parquet")
    df_f1["fever-all-MiniLM-L6-v2"]= pd.read_parquet("/Users/fadilahnurimani/Documents/Study/ut/smt 5/artikel ilmiah/code/fever-all-MiniLM-L6-v2-f1.parquet")
    df_f1["fever-mxbai-embed-large-v1"]= pd.read_parquet("/Users/fadilahnurimani/Documents/Study/ut/smt 5/artikel ilmiah/code/fever-mxbai-embed-large-v1-f1.parquet")
    return (df_f1,)


@app.cell
def _(df_f1):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("SNLI Dataset ", "FEVER Dataset")
    )

    # Color mapping
    color_minilm = "blue"
    color_mxbai = "red"

    # SNLI

    fig.add_trace(
        go.Scatter(
            x=df_f1["snli-all-MiniLM-L6-v2"]["index"],
            y=df_f1["snli-all-MiniLM-L6-v2"]["f1"],
            mode='lines',
            name="all-MiniLM-L6-v2",
            line=dict(color=color_minilm),
            legendgroup="MiniLM"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df_f1["snli-mxbai-embed-large-v1"]["index"],
            y=df_f1["snli-mxbai-embed-large-v1"]["f1"],
            mode='lines',
            name="mxbai-embed-large-v1",
            line=dict(color=color_mxbai),
            legendgroup="mxbai"
        ),
        row=1, col=1
    )

    # FEVER

    fig.add_trace(
        go.Scatter(
            x=df_f1["fever-all-MiniLM-L6-v2"]["index"],
            y=df_f1["fever-all-MiniLM-L6-v2"]["f1"],
            mode='lines',
            name="all-MiniLM-L6-v2",
            line=dict(color=color_minilm),
            legendgroup="MiniLM",
            showlegend=False
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=df_f1["fever-mxbai-embed-large-v1"]["index"],
            y=df_f1["fever-mxbai-embed-large-v1"]["f1"],
            mode='lines',
            name="mxbai-embed-large-v1",
            line=dict(color=color_mxbai),
            legendgroup="mxbai",
            showlegend=False
        ),
        row=1, col=2
    )


    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=1, col=2)

    fig.update_yaxes(title_text="Macro F1", row=1, col=1)
    fig.update_yaxes(title_text="Macro F1", row=1, col=2)


    fig.update_layout(
        height=500,
        width=1000,
        legend_title="Embedding Model"
    )

    fig.show()
    return go, make_subplots


@app.cell
def _(pd):
    df_viz = {}
    df_viz["snli-all-MiniLM-L6-v2"]= pd.read_parquet("/Users/fadilahnurimani/Documents/Study/ut/smt 5/artikel ilmiah/code/snli-all-MiniLM-L6-v2-viz.parquet")
    df_viz["snli-mxbai-embed-large-v1"]= pd.read_parquet("/Users/fadilahnurimani/Documents/Study/ut/smt 5/artikel ilmiah/code/snli-mxbai-embed-large-v1-viz.parquet")
    df_viz["fever-all-MiniLM-L6-v2"]= pd.read_parquet("/Users/fadilahnurimani/Documents/Study/ut/smt 5/artikel ilmiah/code/fever-all-MiniLM-L6-v2-viz.parquet")
    df_viz["fever-mxbai-embed-large-v1"]= pd.read_parquet("/Users/fadilahnurimani/Documents/Study/ut/smt 5/artikel ilmiah/code/fever-mxbai-embed-large-v1-viz.parquet")
    return (df_viz,)


@app.cell
def _():
    color_map = {
        "Supports": "#016FB9",
        "NEI": "#9999A1",
        "Refutes": "#EC4E20"
    }

    mean_line_colors = {
        "Supports": "#00E5FF",   # green
        "NEI": "#8A2BE2",        # purple
        "Refutes": "#FFD60A"     # orange
    }
    return color_map, mean_line_colors


@app.cell
def _(color_map, df_f1, df_viz, go, make_subplots, mean_line_colors):
    theta_dict = {}

    for key, dt in df_f1.items():

        gbest_last = dt.iloc[-1]["gbest"]

        theta_dict[key] = {
            "theta_s": gbest_last[0],
            "theta_c": gbest_last[1]
        }

    titles = [
        "SNLI - all-MiniLM-L6-v2",
        "SNLI - mxbai-embed-large-v1",
        "FEVER - all-MiniLM-L6-v2",
        "FEVER - mxbai-embed-large-v1"
    ]

    keys = [
        "snli-all-MiniLM-L6-v2",
        "snli-mxbai-embed-large-v1",
        "fever-all-MiniLM-L6-v2",
        "fever-mxbai-embed-large-v1"
    ]

    figp = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0,
        specs=[
            [{"type": "polar"}, {"type": "polar"}],
            [{"type": "polar"}, {"type": "polar"}]
        ],
        subplot_titles=titles
    )
    for ann in figp. layout.annotations:
        ann.y = ann.y - 0.025

    for i, key in enumerate(keys):

        row = i // 2 + 1
        col = i % 2 + 1

        dt = df_viz[key]

        theta_s_final = theta_dict[key]["theta_s"]
        theta_c_final = theta_dict[key]["theta_c"]

        mean_theta = (
            df_viz[key]
            .groupby("label")
            .agg({"theta_deg": "mean"})
        ).to_dict()["theta_deg"]
        mean_theta = {
            k: mean_theta[k]
            for k in ["Supports", "NEI", "Refutes"]
        }

        for lbl, color in color_map.items():

            subset = dt[dt["label"] == lbl]

            figp.add_trace(
                go.Scatterpolar(
                    r=subset["r"],
                    theta=subset["theta_deg"],
                    mode="markers",
                    marker=dict(
                        color=color,
                        size=4,
                        opacity=0.7
                    ),
                    name=lbl,
                    legendgroup=lbl,
                    showlegend=(i == 0)
                ),
                row=row,
                col=col
            )

        figp.add_trace(
            go.Scatterpolar(
                r=dt["r"].tolist(),
                theta=dt["theta_deg"].tolist(),
                mode="markers",
                marker=dict(
                    color=[color_map[l] for l in dt["label"]],
                    size=4
                ),
                showlegend=False
            ),
            row=row,
            col=col
        )

        figp.add_trace(
            go.Scatterpolar(
                r=[0, 1.05],
                theta=[theta_s_final, theta_s_final],
                mode="lines",
                line=dict(
                    color="#016FB9",
                    width=2,
                    dash="dash"
                ),
                name="θ_s",
                legendgroup="theta_s",
                showlegend=(i == 0)
            ),
            row=row,
            col=col
        )

        figp.add_trace(
            go.Scatterpolar(
                r=[0, 1.05],
                theta=[theta_c_final, theta_c_final],
                mode="lines",
                line=dict(
                    color="#EC4E20",
                    width=2,
                    dash="dash"
                ),
                name="θ_r",
                legendgroup="theta_c",
                showlegend=(i == 0)
            ),
            row=row,
            col=col
        )

        for lbl, theta_mean in mean_theta.items():

            figp.add_trace(
                go.Scatterpolar(
                    r=[0.95],
                    theta=[theta_mean],
                    mode="markers",
                    marker=dict(
                        symbol="star",
                        size=20,
                        color=mean_line_colors[lbl]
                    ),
                    name=f"Mean θ ({lbl})",
                    legendgroup=f"mean_{lbl}",
                    showlegend=(i == 0)
                ),
                row=row,
                col=col
            )

    

    figp.update_layout(
        height=700,
        width=1100,
        legend_title="Category",
    )
    figp.update_layout(
        legend=dict(
            x=1.05,
            y=0.95
        )
    )

    figp.update_polars(
        sector=[0, 180],
        radialaxis=dict(range=[0, 1])
    )

    figp.show()
    return


if __name__ == "__main__":
    app.run()
