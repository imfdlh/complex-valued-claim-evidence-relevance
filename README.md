# Complex-Valued Claim-Evidence Relevance

Implementation of the paper:

"Complex-Valued Representation of Relevance and Semantic Relations for Claim-Evidence Pairs with PSO Boundary Calibration"

## Paper

**Complex-Valued Representation of Relevance and Semantic Relations for Claim-Evidence Pairs with PSO Boundary Calibration**

Published in Computer Engineering and Applications Journal (ComEngApp).

🔗 Paper: https://comengapp.unsri.ac.id/index.php/comengapp/article/view/1421

## Environment

- Python 3.12.7
- [Marimo](https://marimo.io/)
- Docker
- Dependencies: `requirements.txt`

## Reproducibility

Build the Docker image:

```bash
docker build -t complex-valued-claim-evidence-relevance .
```

Run the container:

```bash
docker run -it --rm -p 2718:2718 complex-valued-claim-evidence-relevance bash
```

Open a Marimo notebook:

```bash
marimo edit notebooks/Result_summary.py --host 0.0.0.0 --port 2718
```

Then open [http://localhost:2718](http://localhost:2718) in your browser.

Other experiment notebooks can be opened using the same command.
