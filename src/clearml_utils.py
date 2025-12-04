import os
from typing import Any, Dict, Optional


def init_task(
    enabled: bool,
    project: str,
    task_name: str,
    tags=None,
    params: Optional[Dict[str, Any]] = None,
):
    task = None
    if enabled:
        try:
            from clearml import Task

            task = Task.init(
                project_name=project,
                task_name=task_name,
                tags=tags or [],
                reuse_last_task_id=False,
                auto_connect_arg_parser=False,
            )
            if params:
                task.connect(params)
        except Exception as error:
            print(
                f"[ClearML] Failed to init task: {error}. Proceeding without ClearML."
            )
            task = None
    return task


def log_scalar(task, title: str, series: str, value: float, step: int):
    if task is None:
        return
    try:
        task.get_logger().report_scalar(
            title=title, series=series, value=value, iteration=step
        )
    except Exception as error:
        print(f"[ClearML] log_scalar error: {error}")


def log_figure(task, title: str, series: str, figure, step: int = 0):
    if task is None:
        return
    try:
        task.get_logger().report_matplotlib_figure(
            title=title, series=series, figure=figure, iteration=step
        )
    except Exception as error:
        print(f"[ClearML] log_figure error: {error}")


def upload_model(task, local_path: str, name: str = "best.pt"):
    if task is None:
        return
    try:
        task.update_output_model(
            model_path=local_path, name=name, auto_delete_file=False
        )
    except Exception as error:
        print(f"[ClearML] upload_model error: {error}")


def log_image(task, title: str, image_path: str):
    """Log an image to ClearML"""
    if task is None:
        return
    try:
        task.get_logger().report_image(title=title, series=title, local_path=image_path)
    except Exception as error:
        print(f"[ClearML] log_image error for '{title}' at '{image_path}': {error}")
