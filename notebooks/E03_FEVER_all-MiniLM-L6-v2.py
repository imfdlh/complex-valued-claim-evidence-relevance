import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    # Load Dataset
    import pandas as pd
    df = pd.read_parquet(
        "/Users/fadilahnurimani/Documents/Study/ut/smt 5/artikel ilmiah/code/sample_9000_fevernli.parquet"
    )
    print(df.shape)
    print(df['label'].value_counts())

    label_map = {"SUPPORTS": +1, "NOT ENOUGH INFO": 0, "REFUTES": -1}

    EVIDENCES = [
        {
            "claim": row["hypothesis"],
            "text":  row["premise"],
            "y":     row["y"],
        }
        for _, row in df.iterrows()
    ]
    return EVIDENCES, df, pd


@app.cell
def _(df):
    df
    return


@app.cell
def _(EVIDENCES):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    import json
    import cmath
    import math
    import csv
    import os
    from sklearn.model_selection import train_test_split

    np_seed = 123

    # Initialize Model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def get_embedding(text, backend="hf"):
        if backend == "hf":
            return model.encode(text, normalize_embeddings=True)

    # Compute Z for each claim-evidence
    def compute_features(evidences, backend="hf"):
        results = []
        for e in evidences:
            c_emb = get_embedding(e["claim"],  backend)
            e_emb = get_embedding(e["text"],   backend)

            cos_sim   = float(np.dot(c_emb, e_emb))
            cos_sim   = np.clip(cos_sim, -1.0, 1.0)

            theta_rad = np.arccos(cos_sim)
            theta_deg = np.degrees(theta_rad)

            r = abs(cos_sim)
            Z = r * np.exp(1j * theta_rad)
            e_i_theta = np.exp(1j * theta_rad)

            k = theta_rad / np.pi
            polar_form = f"{r:.4f} e^(i {k:.4f}π)"

            results.append({
                "claim":     e["claim"],
                "text":      e["text"],
                "y":         e["y"],
                "cosine":    cos_sim,
                "r":         r,
                "theta_deg": theta_deg,
                "r, theta": (cmath.polar(Z)[0], math.degrees(cmath.polar(Z)[1])),
                "e^(u*theta)": e_i_theta,
                "polar": polar_form,
                "Re":        Z.real,
                "Im":        Z.imag,
                "Z":         Z,
            })

            # Save results
        with open("/Users/fadilahnurimani/Documents/Study/ut/smt 5/artikel ilmiah/data/results_fever_all-MiniLM-L6-v2.json", "w") as f:
            json.dump([{
                    "claim":     r["claim"],
                    "text":      r["text"],
                    "y":         r["y"],
                    "cosine":    float(r["cosine"]),
                    "r":         float(r["r"]),
                    "theta_deg": float(r["theta_deg"]),
                    "Re":        float(r["Re"]),
                    "Im":        float(r["Im"]),
            } for r in results], f, indent=2)
        return results


    # Predict label based on theta
    def predict_score(res, theta_s, theta_c):
        theta = res["theta_deg"]
        if theta <= theta_s:
            return +1
        elif theta >= theta_c:
            return -1
        else:
            return  0

    # Objective function
    def objective(params, results, return_detail=False):
        theta_s, theta_c = params

        if theta_s >= theta_c or theta_s < 0 or theta_c > 180:
            if return_detail:
                return 1e9, None
            return 1e9

        y_true, y_pred = [], []
        for r in results:
            pred = predict_score(r, theta_s, theta_c)
            y_true.append(r["y"])
            y_pred.append(pred)

        mapping = {-1: 0, 0: 1, 1: 2}
        yt = [mapping[y] for y in y_true]
        yp = [mapping[y] for y in y_pred]

        f1_macro = f1_score(yt, yp, average="macro", zero_division=0)

        if return_detail:
            f1_per_class = f1_score(yt, yp, average=None, zero_division=0)
            precision    = precision_score(yt, yp, average=None, zero_division=0)
            recall       = recall_score(yt, yp, average=None, zero_division=0)
            accuracy = accuracy_score(yt, yp)
            return -f1_macro, {
                "f1_per_class": f1_per_class,
                "precision":    precision,
                "recall":       recall,
                "accuracy":    accuracy,
            }

        return -f1_macro

    # Optimize using PSO
    def pso_optimize(results, n_particles=30, n_iter=200):
        np.random.seed(np_seed)
        pos = np.column_stack([
            np.random.uniform(0,  70, n_particles),   # theta_s
            np.random.uniform(60, 100, n_particles),  # theta_c — based on data
        ])
        vel = np.zeros_like(pos)

        pbest     = pos.copy()
        pbest_val = np.array([objective(p, results) for p in pos])

        gbest     = pbest[np.argmin(pbest_val)].copy()
        gbest_val = pbest_val.min()

        # Log history
        history = {
            "gbest":  [gbest.tolist()],
            "f1":     [-gbest_val],
            "particles": [pos.tolist()],
        }

        for iter_num in range(n_iter):
            for i in range(n_particles):
                r1, r2 = np.random.rand(), np.random.rand()

                vel[i] = (
                    0.9 * vel[i]
                    + 2.0 * r1 * (pbest[i] - pos[i])
                    + 2.0 * r2 * (gbest    - pos[i])
                )

                pos[i]    += vel[i]
                pos[i, 0]  = np.clip(pos[i, 0],   0,  90)
                pos[i, 1] = np.clip(pos[i, 1], 50, 180)

                val = objective(pos[i], results)
                if val < pbest_val[i]:
                    pbest[i]     = pos[i].copy()
                    pbest_val[i] = val

            best_idx = np.argmin(pbest_val)
            if pbest_val[best_idx] < gbest_val:
                gbest     = pbest[best_idx].copy()
                gbest_val = pbest_val[best_idx]

            if iter_num % 10 == 0:
                print(f"Iter {iter_num:3d}: F1={-gbest_val:.4f}, "
                      f"θ_s={gbest[0]:.2f}°, θ_c={gbest[1]:.2f}°")

            history["gbest"].append(gbest.tolist())
            history["f1"].append(-gbest_val)
            history["particles"].append(pos.tolist())

        # Log history
        with open("/Users/fadilahnurimani/Documents/Study/ut/smt 5/artikel ilmiah/data/pso_history_fever_all-MiniLM-L6-v2.json", "w") as f:
            json.dump(history, f)

        return gbest, -gbest_val

    # Run process
    def run():

        print("Compute embedding and Z for each pairing")
        results = compute_features(EVIDENCES)

        # Split data into calibration and held-out test subsets
        train_results, test_results = train_test_split(
            results,
            test_size=0.2,
            random_state=np_seed,
            stratify=[r["y"] for r in results]
        )

        print(f"\nPairing count: {len(results)}")
        print(f"Calibration set: {len(train_results)}")
        print(f"Test set      : {len(test_results)}")

        print(
            f"Calibration label distribution: "
            f"support={sum(r['y']==+1 for r in train_results)}, "
            f"nei={sum(r['y']==0 for r in train_results)}, "
            f"refute={sum(r['y']==-1 for r in train_results)}"
        )

        print(
            f"Test label distribution: "
            f"support={sum(r['y']==+1 for r in test_results)}, "
            f"nei={sum(r['y']==0 for r in test_results)}, "
            f"refute={sum(r['y']==-1 for r in test_results)}"
        )

        print("\n=== Result of Z (COMPLEX) ===\n")
        for r in results:
            print(f"Premise    : {r['claim'][:60]}...")
            print(f"Hypothesis : {r['text'][:60]}...")
            print(f"Label      : {r['y']}")
            print(f"Cosine     : {r['cosine']:.4f}")
            print(f"r          : {r['r']:.4f}")
            print(f"θ          : {r['theta_deg']:.2f}°")
            print(f"r, theta   : {r['r, theta']}")
            print(f"e^(u*theta): {r['e^(u*theta)']:.4f}")
            print(f"Polar form : {r['polar']}")
            print(f"Re(Z)      : {r['Re']:.4f}")
            print(f"Im(Z)      : {r['Im']:.4f}")
            print(f"Z          : {r['Z']:.4f}")
            print("-" * 60)

        # ---------------------------------------------------------
        # PSO calibration on calibration set only
        # ---------------------------------------------------------
        print("\nRun PSO optimization on calibration set...")
        best_params, train_f1 = pso_optimize(train_results)

        theta_s, theta_c = best_params

        # ---------------------------------------------------------
        # Evaluate the calibrated boundaries on calibration set
        # ---------------------------------------------------------
        _, train_detail = objective(
            [theta_s, theta_c],
            train_results,
            return_detail=True
        )

        # ---------------------------------------------------------
        # Evaluate the same fixed boundaries on held-out test set
        # ---------------------------------------------------------
        _, test_detail = objective(
            [theta_s, theta_c],
            test_results,
            return_detail=True
        )

        test_f1_per_class = test_detail["f1_per_class"]
        test_precision    = test_detail["precision"]
        test_recall       = test_detail["recall"]
        test_accuracy     = test_detail["accuracy"]

        train_f1_per_class = train_detail["f1_per_class"]
        train_precision    = train_detail["precision"]
        train_recall       = train_detail["recall"]
        train_accuracy     = train_detail["accuracy"]

        # Compute test macro-F1 explicitly
        y_true = [r["y"] for r in test_results]
        y_pred = [
            predict_score(r, theta_s, theta_c)
            for r in test_results
        ]

        mapping = {-1: 0, 0: 1, 1: 2}

        yt = [mapping[y] for y in y_true]
        yp = [mapping[y] for y in y_pred]

        test_f1 = f1_score(
            yt,
            yp,
            average="macro",
            zero_division=0
        )

        print("\n=== PSO Result ===")
        print(f"θ_s (support boundary) : {theta_s:.2f}°")
        print(f"θ_r (refute boundary)  : {theta_c:.2f}°")
        print(f"Train F1-macro         : {train_f1:.5f}")
        print(f"Test F1-macro          : {test_f1:.5f}")

        label_names = [
            "Refutes (-1)",
            "NEI (0)",
            "Supports (+1)"
        ]

        # ---------------------------------------------------------
        # Calibration set results
        # ---------------------------------------------------------
        print("\n=== Calibration Set Results ===")
        print(f"{'Class':<22} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 55)

        for i, name in enumerate(label_names):
            print(
                f"{name:<22} "
                f"{train_precision[i]:>10.5f} "
                f"{train_recall[i]:>10.5f} "
                f"{train_f1_per_class[i]:>10.5f}"
            )

        print("-" * 55)
        print(
            f"{'Macro Average':<22} "
            f"{train_precision.mean():>10.5f} "
            f"{train_recall.mean():>10.5f} "
            f"{train_f1:>10.5f}"
        )
        print(f"Accuracy: {train_accuracy:>10.5f}")

        # ---------------------------------------------------------
        # Held-out test set results
        # ---------------------------------------------------------
        print("\n=== Held-Out Test Results ===")
        print(f"{'Class':<22} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 55)

        for i, name in enumerate(label_names):
            print(
                f"{name:<22} "
                f"{test_precision[i]:>10.5f} "
                f"{test_recall[i]:>10.5f} "
                f"{test_f1_per_class[i]:>10.5f}"
            )

        print("-" * 55)
        print(
            f"{'Macro Average':<22} "
            f"{test_precision.mean():>10.5f} "
            f"{test_recall.mean():>10.5f} "
            f"{test_f1:>10.5f}"
        )
        print(f"Accuracy: {test_accuracy:>10.5f}")

        # ---------------------------------------------------------
        # Semantic relation zones
        # ---------------------------------------------------------
        print("\nInterpretation:")
        print(f"  Supports    : θ ∈ [0°,     {theta_s:.1f}°]")
        print(f"  NEI         : θ ∈ ({theta_s:.1f}°,  {theta_c:.1f}°)")
        print(f"  Refutes     : θ ∈ [{theta_c:.1f}°, 180°]")

        # ---------------------------------------------------------
        # Classification results on held-out test set
        # ---------------------------------------------------------
        print("\n=== Held-Out Test Classification Results ===\n")

        correct = 0

        for r in test_results:
            pred = predict_score(r, theta_s, theta_c)

            label = {
                +1: "Supports",
                 0: "NEI",
                -1: "Refutes"
            }[pred]

            true = {
                +1: "Supports",
                 0: "NEI",
                -1: "Refutes"
            }[r["y"]]

            ok = pred == r["y"]
            correct += ok

        print(
            f"\nAccuracy: {correct}/{len(test_results)} "
            f"= {correct / len(test_results) * 100:.1f}%"
        )

        # ---------------------------------------------------------
        # θ distribution statistics
        # Keep these based on all available pairs for descriptive analysis
        # ---------------------------------------------------------
        import statistics

        for label, y_val in [
            ("Supports", +1),
            ("NEI", 0),
            ("Refutes", -1)
        ]:
            thetas = [
                r["theta_deg"]
                for r in results
                if r["y"] == y_val
            ]

            print(
                f"{label}: "
                f"mean={statistics.mean(thetas):.1f}°  "
                f"std={statistics.stdev(thetas):.1f}°  "
                f"min={min(thetas):.1f}°  "
                f"max={max(thetas):.1f}°"
            )


    if __name__ == "__main__":
        run()
    return json, np


@app.cell
def _(json, pd):
    with open("/Users/fadilahnurimani/Documents/Study/ut/smt 5/artikel ilmiah/data/results_fever_all-MiniLM-L6-v2.json") as f:
        results = json.load(f)

    label_map_text  = {1: "Supports", 0: "NEI", -1: "Refutes"}
    import plotly.express as px

    colors = px.colors.qualitative.Plotly

    def get_class(y):
        return label_map_text[y]

    df_viz = pd.DataFrame(results)
    df_viz['label'] = df_viz['y'].apply(lambda x: get_class(x))
    df_viz
    return df_viz, results


@app.cell
def _(json):
    with open("/Users/fadilahnurimani/Documents/Study/ut/smt 5/artikel ilmiah/data/pso_history_fever_all-MiniLM-L6-v2.json") as r:
        history = json.load(r)

    theta_s_final = history["gbest"][-1][0]
    theta_c_final = history["gbest"][-1][1]
    return history, theta_c_final, theta_s_final


@app.cell
def _():
    color_map = {
        "Supports": "#016FB9",
        "NEI": "#9999A1",
        "Refutes": "#EC4E20"
    }
    return (color_map,)


@app.cell
def _(color_map, df_viz, theta_c_final, theta_s_final):
    import plotly.graph_objects as go

    fig = go.Figure()

    for lbl, col in color_map.items():
        subset = df_viz[df_viz["label"] == lbl]

        r_vals     = []
        theta_vals = []

        for _, row in subset.iterrows():
            r_vals     += [0, row["r"],         None]
            theta_vals += [row["theta_deg"], row["theta_deg"], None]

        fig.add_trace(go.Scatterpolar(
            r          = r_vals,
            theta      = theta_vals,
            mode       = "lines",
            name       = lbl,
            line       = dict(color=col, width=1),
            opacity    = 0.4,
        ))

    fig.add_trace(go.Scatterpolar(
        r     = df_viz["r"].tolist(),
        theta = df_viz["theta_deg"].tolist(),
        mode  = "markers",
        marker = dict(
            color  = [color_map[l] for l in df_viz["label"]],
            size   = 4,
        ),
        showlegend = False,
    ))

    for theta_val, col, name in [
        (theta_s_final, "#016FB9", f"θ_s*={theta_s_final:.1f}°"),
        (theta_c_final, "#EC4E20", f"θ_c*={theta_c_final:.1f}°"),
    ]:
        fig.add_trace(go.Scatterpolar(
            r=[0, 1.05], theta=[theta_val, theta_val],
            mode="lines",
            line=dict(color=col, width=2, dash="dash"),
            name=name,
        ))

    fig.update_layout(
        polar=dict(
            sector=[0, 180],
            radialaxis=dict(range=[0, 1]),
        )
    )
    fig.show()
    return


@app.cell
def _(pd, results):
    pd.DataFrame(results).groupby(["y"])[["claim","text","theta_deg"]].describe()
    return


@app.cell
def _(pd, results):
    pd.DataFrame(results).groupby(["y"])[["claim","text","r"]].describe()
    return


@app.cell
def _(history, pd):
    df_f1 = pd.DataFrame(history)[["f1", "gbest"]].reset_index()
    df_f1.to_parquet("/Users/fadilahnurimani/Documents/Study/ut/smt 5/artikel ilmiah/code/fever-all-MiniLM-L6-v2-f1.parquet")
    df_f1
    return


@app.cell
def _(df_viz):
    df_viz.to_parquet("/Users/fadilahnurimani/Documents/Study/ut/smt 5/artikel ilmiah/code/fever-all-MiniLM-L6-v2-viz.parquet")
    return


@app.cell
def _(df_viz):
    df_viz["label"].value_counts()
    return


@app.cell
def _(df, np):
    df['hypothesis'].replace('', np.nan).isnull().sum()
    return


@app.cell
def _(df, np):
    df['premise'].replace('', np.nan).isnull().sum()
    return


if __name__ == "__main__":
    app.run()
