from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SimpleDropoutDQN(nn.Module):
    def __init__(self, state_size: int, action_size: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(dropout * 0.5)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        return self.fc3(x)


class SimpleResidualDQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out)) + out
        return self.fc3(out)


class DuelingDQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + (a - a.mean(dim=1, keepdim=True))


class LSTMDQN(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(state_size, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, action_size))

    def forward(self, x: torch.Tensor, hidden=None) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x, hidden)
        return self.fc(out[:, -1, :])


class AttentionDQN(nn.Module):
    def __init__(self, state_size: int, action_size: int, embed: int = 128, heads: int = 4):
        super().__init__()
        self.embed = nn.Linear(state_size, embed)
        self.attn = nn.MultiheadAttention(embed, heads, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(embed, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        e = self.embed(x)
        a, _ = self.attn(e, e, e)
        return self.fc((e + a).mean(dim=1))


MODEL_CLASSES = {
    "simple_dqn": SimpleDQN,
    "dropout_dqn": SimpleDropoutDQN,
    "residual_dqn": SimpleResidualDQN,
    "dueling_dqn": DuelingDQN,
    "lstm_dqn": LSTMDQN,
    "attention_dqn": AttentionDQN,
}
ACTION_MAP = {0: "HOLD", 1: "BUY", 2: "SELL"}


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_stock_data(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    else:
        df.index = pd.to_datetime(df.index, errors="coerce")
    return df.sort_index()


@st.cache_resource(show_spinner=False)
def load_models(project_root: str) -> tuple[dict, dict, dict]:
    root = Path(project_root)
    models_dir = root / "models"

    model_configs = load_json(str(models_dir / "model_configs.json"))
    feature_groups = load_json(str(models_dir / "feature_groups.json"))
    ensemble_cfg = load_json(str(models_dir / "ensemble_config.json"))

    model_paths = ensemble_cfg.get("model_paths", {})
    loaded = {}

    for model_name, cfg in model_configs.items():
        rel = model_paths.get(model_name, f"models/{model_name}_v2.pt")
        checkpoint = root / rel
        if not checkpoint.exists():
            continue

        arch = cfg.get("architecture", model_name)
        model_cls = MODEL_CLASSES.get(arch)
        if model_cls is None:
            continue

        state_dim = int(cfg.get("state_dim", 11))
        action_dim = int(cfg.get("action_dim", 3))

        model = model_cls(state_dim, action_dim)
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        loaded[model_name] = model

    return loaded, model_configs, feature_groups


def build_state(row: pd.Series, feature_columns: list[str]) -> np.ndarray:
    market = row[feature_columns].astype(float).values
    portfolio = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return np.concatenate([market, portfolio]).astype(np.float32)


def get_prediction(model: nn.Module, state: np.ndarray) -> tuple[int, np.ndarray]:
    with torch.no_grad():
        tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q = model(tensor).squeeze(0).cpu().numpy()
    action = int(np.argmax(q))
    return action, q


def main() -> None:
    st.set_page_config(page_title="Stock Dashboard", layout="wide")
    st.title("Stock Analysis")

    # Automatically set project root (no sidebar input)
    root = Path(__file__).resolve().parent

    # use `root` normally below
    st.write(f"Project Root: {root}")

    data_dir = root / "data" / "processed"
    models_dir = root / "models"
    if not data_dir.exists() or not models_dir.exists():
        st.error("Invalid project root. Expected folders: data/processed and models")
        st.stop()

    files = sorted(data_dir.glob("*_*.parquet"))
    tickers = sorted({f.stem.split("_")[0] for f in files if "_" in f.stem})
    splits = ["train", "val", "test"]

    if not tickers:
        st.error("No processed parquet files found.")
        st.stop()

    ticker = st.sidebar.selectbox("Ticker", tickers, index=0)
    split = st.sidebar.selectbox("Split", splits, index=2)
    parquet_path = data_dir / f"{ticker}_{split}.parquet"

    if not parquet_path.exists():
        st.error(f"Missing file: {parquet_path}")
        st.stop()

    df = load_stock_data(str(parquet_path))
    models, model_configs, feature_groups = load_models(str(root))

    if not models:
        st.error("No models could be loaded from models/*.pt")
        st.stop()

    available_models = list(models.keys())
    selected_models = st.sidebar.multiselect(
        "Models",
        available_models,
        default=available_models,
    )

    if not selected_models:
        st.warning("Select at least one model.")
        st.stop()

    row_idx = st.sidebar.slider("Row index for prediction", 0, len(df) - 1, len(df) - 1)
    row = df.iloc[row_idx]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Last Close", f"{df['Close'].iloc[-1]:.2f}")
    col3.metric("Mean Return", f"{df['Returns'].mean() * 100:.2f}%")
    col4.metric("Volatility", f"{df['Returns'].std() * 100:.2f}%")

    chart_df = df.reset_index().rename(columns={df.index.name or "index": "Date"})
    fig_price = px.line(chart_df, x="Date", y="Close", title=f"{ticker} Close Price ({split})")
    st.plotly_chart(fig_price, use_container_width=True)

    returns_fig = px.histogram(df, x="Returns", nbins=40, title="Return Distribution")
    st.plotly_chart(returns_fig, use_container_width=True)

    st.subheader("Prediction")
    st.caption(f"Using row {row_idx} | Date: {df.index[row_idx].date() if pd.notna(df.index[row_idx]) else 'N/A'}")

    pred_rows = []
    q_values_all = []
    action_votes = []

    for name in selected_models:
        cfg = model_configs.get(name, {})
        group = cfg.get("feature_group", "all")
        feature_cols = feature_groups.get(group, [])

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            st.warning(f"Skipping {name}: missing columns {missing}")
            continue

        state = build_state(row, feature_cols)
        action, q_values = get_prediction(models[name], state)
        q_values_all.append(q_values)
        action_votes.append(action)

        pred_rows.append(
            {
                "model": name,
                "feature_group": group,
                "action": ACTION_MAP[action],
                "q_hold": round(float(q_values[0]), 4),
                "q_buy": round(float(q_values[1]), 4),
                "q_sell": round(float(q_values[2]), 4),
            }
        )

    if not pred_rows:
        st.error("No predictions generated. Check feature columns and model files.")
        st.stop()

    pred_df = pd.DataFrame(pred_rows)
    st.dataframe(pred_df, use_container_width=True)

    vote_counts = np.bincount(action_votes, minlength=3)
    hard_action = int(np.argmax(vote_counts))
    mean_q = np.mean(np.vstack(q_values_all), axis=0)
    soft_action = int(np.argmax(mean_q))

    e1, e2 = st.columns(2)
    e1.metric("Hard Vote", ACTION_MAP[hard_action])
    e2.metric("Soft Vote", ACTION_MAP[soft_action])

    st.write(
        {
            "vote_counts": {
                "HOLD": int(vote_counts[0]),
                "BUY": int(vote_counts[1]),
                "SELL": int(vote_counts[2]),
            },
            "mean_q": {
                "HOLD": round(float(mean_q[0]), 4),
                "BUY": round(float(mean_q[1]), 4),
                "SELL": round(float(mean_q[2]), 4),
            },
        }
    )


if __name__ == "__main__":
    main()
