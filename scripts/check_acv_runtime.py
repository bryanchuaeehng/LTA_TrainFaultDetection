"""Create a parity report without retraining; run in source and target runtimes."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from src.acv.predict import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    report = {"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__, "cases": {}}
    for path in args.inputs:
        result = predict(path)
        ranking = pd.DataFrame(result.attrs["diagnostics"]["ranking"]).sort_values("car")
        report["cases"][path.name] = {
            "ranked_cars": result.iloc[0].ranked_cars,
            "columns": list(ranking),
            "values": ranking.drop(columns="car").to_numpy().tolist(),
        }
        print(path.name, result.iloc[0].ranked_cars, flush=True)
    if args.compare:
        previous = json.loads(args.compare.read_text())
        assert set(previous["cases"]) == set(report["cases"])
        for name, current in report["cases"].items():
            original = previous["cases"][name]
            assert original["ranked_cars"] == current["ranked_cars"]
            assert original["columns"] == current["columns"]
            np.testing.assert_allclose(original["values"], current["values"], rtol=1e-12, atol=1e-12)
        report["parity_against"] = previous["sklearn"]
        report["parity"] = "all six case rankings and component scores agree to 1e-12"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
