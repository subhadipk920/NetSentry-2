"""
Serving Predictor Service (`netsentry.serving.predictor`).
---------------------------------------------------------
Applies champion model inference using the champion model's frozen threshold.
Supports DataFrame and Dictionary feature inputs.
"""

from typing import Any, Dict, Tuple, Union
import numpy as np
import polars as pl

from netsentry.features.engineer import engineer_network_features
from netsentry.models.interface import has_predict_proba
from netsentry.serving.model_loader import ChampionModel


EXPECTED_FEATURE_ORDER = [
    'Destination_Port', 'Protocol', 'Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets',
    'Total_Length_of_Fwd_Packets', 'Total_Length_of_Bwd_Packets', 'Fwd_Packet_Length_Max', 'Fwd_Packet_Length_Min',
    'Fwd_Packet_Length_Mean', 'Fwd_Packet_Length_Std', 'Bwd_Packet_Length_Max', 'Bwd_Packet_Length_Min',
    'Bwd_Packet_Length_Mean', 'Bwd_Packet_Length_Std', 'Flow_Bytes_s', 'Flow_Packets_s', 'Flow_IAT_Mean',
    'Flow_IAT_Std', 'Flow_IAT_Max', 'Flow_IAT_Min', 'Fwd_IAT_Total', 'Fwd_IAT_Mean', 'Fwd_IAT_Std',
    'Fwd_IAT_Max', 'Fwd_IAT_Min', 'Bwd_IAT_Total', 'Bwd_IAT_Mean', 'Bwd_IAT_Std', 'Bwd_IAT_Max',
    'Bwd_IAT_Min', 'Fwd_PSH_Flags', 'Fwd_URG_Flags', 'Fwd_Header_Length', 'Bwd_Header_Length',
    'Fwd_Packets_s', 'Bwd_Packets_s', 'Min_Packet_Length', 'Max_Packet_Length', 'Packet_Length_Mean',
    'Packet_Length_Std', 'Packet_Length_Variance', 'FIN_Flag_Count', 'SYN_Flag_Count', 'RST_Flag_Count',
    'PSH_Flag_Count', 'ACK_Flag_Count', 'URG_Flag_Count', 'CWE_Flag_Count', 'ECE_Flag_Count',
    'Down_Up_Ratio', 'Average_Packet_Size', 'Init_Win_bytes_forward',
    'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward', 'Active_Mean', 'Active_Std',
    'Active_Max', 'Active_Min', 'Idle_Mean', 'Idle_Std', 'Idle_Max', 'Idle_Min',
    'Fwd_to_Bwd_Packet_Ratio', 'Fwd_to_Bwd_Byte_Ratio', 'Avg_Fwd_Packet_Payload', 'SYN_no_ACK_Indicator',
    'RST_Density', 'Log_Flow_Duration', 'Flow_Packet_Density'
]


class Predictor:
    """Coordinates prediction execution using the loaded ChampionModel."""

    def __init__(self, champion_model: ChampionModel):
        self.champion = champion_model

    def _prepare_matrix(self, features: Union[Dict[str, float], pl.DataFrame, np.ndarray]) -> np.ndarray:
        """Ensures incoming feature sets are strictly aligned to the exact 71-feature training schema."""
        if isinstance(features, dict):
            feat_dict = dict(features)
            # Guarantee Protocol exists (default TCP = 6.0)
            if "Protocol" not in feat_dict:
                feat_dict["Protocol"] = 6.0

            df = pl.DataFrame([feat_dict])
            # Auto-compute 7 engineered features if not already present
            if "Flow_Packet_Density" not in feat_dict:
                df = engineer_network_features(df)

            # Pad any missing columns from EXPECTED_FEATURE_ORDER with 0.0
            for col in EXPECTED_FEATURE_ORDER:
                if col not in df.columns:
                    df = df.with_columns(pl.lit(0.0).alias(col))

            # Strictly select ONLY the 71 expected features in exact canonical order
            X = df.select(EXPECTED_FEATURE_ORDER).to_numpy()

        elif isinstance(features, pl.DataFrame):
            df = features
            if "Protocol" not in df.columns:
                df = df.with_columns(pl.lit(6.0).alias("Protocol"))
            if "Flow_Packet_Density" not in df.columns:
                df = engineer_network_features(df)
            for col in EXPECTED_FEATURE_ORDER:
                if col not in df.columns:
                    df = df.with_columns(pl.lit(0.0).alias(col))
            X = df.select(EXPECTED_FEATURE_ORDER).to_numpy()

        else:
            X = np.asarray(features)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            # If shape is greater than 71, take first 71
            if X.shape[1] > len(EXPECTED_FEATURE_ORDER):
                X = X[:, :len(EXPECTED_FEATURE_ORDER)]

        return X

    def predict(self, features: Union[Dict[str, float], pl.DataFrame, np.ndarray]) -> Tuple[int, float]:
        """
        Executes binary threat prediction.
        Returns (prediction, probability).
        """
        X = self._prepare_matrix(features)

        model = self.champion.model
        threshold = self.champion.threshold

        if has_predict_proba(model):
            probs = model.predict_proba(X)
            attack_prob = float(probs[0, 1]) if probs.ndim == 2 and probs.shape[1] == 2 else float(probs[0])
            prediction = int(attack_prob >= threshold)
        else:
            preds = model.predict(X)
            prediction = int(preds[0])
            attack_prob = float(prediction)

        return prediction, round(attack_prob, 4)
