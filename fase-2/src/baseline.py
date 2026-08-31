from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data import build_pipeline, load_data

RANDOM_STATE = 42
ALGORITHMS = ("Regressao Logistica", "Arvore de Decisao", "Random Forest")


def _build_models(base_steps: list) -> dict[str, Pipeline]:
    return {
        "Regressao Logistica": Pipeline(
            base_steps + [("clf", LogisticRegression(max_iter=10000, random_state=RANDOM_STATE))]
        ),
        "Arvore de Decisao": Pipeline(
            base_steps + [("clf", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE))]
        ),
        "Random Forest": Pipeline(
            base_steps
            + [("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))]
        ),
    }


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


def train_baseline_models() -> dict[str, dict]:
    """Trains the 3 Modulo 1 models exactly as analysis.ipynb does."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    models = _build_models(build_pipeline().steps)

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = {
            "model": model,
            "metrics": evaluate(model, X_test, y_test),
        }
    return results


def select_champion(results: dict[str, dict]) -> tuple[str, dict]:
    champion_name = max(results, key=lambda n: results[n]["metrics"]["f1"])
    return champion_name, results[champion_name]
