import pandas as pd


def engineer_features(df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()

    for col in ["vibration", "temperature", "pressure"]:
        df[f"{col}_rolling_mean"] = df[col].rolling(window).mean()
        df[f"{col}_rolling_std"] = df[col].rolling(window).std()
        df[f"{col}_delta"] = df[col].diff()
        df[f"{col}_zscore"] = (
            (df[col] - df[col].rolling(window).mean())
            / df[col].rolling(window).std()
        )

    df["hours_since_last_spike"] = (
        df["vibration"]
        .gt(df["vibration"].mean() + 2 * df["vibration"].std())
        .cumsum()
    )

    return df.dropna()


# rolling mean/std - behavior over 24 hours
# delta - change in value from previous reading
# zscore - how many standard deviations from the mean
# hours_since_last_spike - how long since the last spike