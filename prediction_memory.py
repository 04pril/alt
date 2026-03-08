from __future__ import annotations

import prediction_store

__all__ = [
    "attach_order_to_prediction",
    "filter_prediction_history",
    "load_prediction_log",
    "load_model_registry",
    "prediction_id_for_run",
    "refresh_prediction_actuals",
    "save_prediction_snapshot",
    "summarize_prediction_accuracy",
]


def save_prediction_snapshot(*args, **kwargs):
    return prediction_store.save_prediction_snapshot(*args, **kwargs)


def refresh_prediction_actuals(*args, **kwargs):
    return prediction_store.refresh_prediction_actuals(*args, **kwargs)


def load_prediction_log(*args, **kwargs):
    return prediction_store.load_prediction_log(*args, **kwargs)


def filter_prediction_history(*args, **kwargs):
    return prediction_store.filter_prediction_history(*args, **kwargs)


def summarize_prediction_accuracy(*args, **kwargs):
    return prediction_store.summarize_prediction_accuracy(*args, **kwargs)


def prediction_id_for_run(*args, **kwargs):
    return prediction_store.prediction_id_for_run(*args, **kwargs)


def attach_order_to_prediction(*args, **kwargs):
    return prediction_store.attach_order_to_prediction(*args, **kwargs)


def load_model_registry(*args, **kwargs):
    return prediction_store.load_model_registry(*args, **kwargs)
