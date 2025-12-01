from typing import Any, Dict, Optional
import os

def init_task(enabled: bool, project: str, task_name: str, tags=None, params: Optional[Dict[str, Any]] = None):
    task = None
    if enabled:
        try:
            from clearml import Task
            task = Task.init(project_name=project, task_name=task_name, tags=tags or [])
            if params:
                task.connect(params)
        except Exception as e:
            print(f"[ClearML] Failed to init task: {e}. Proceeding without ClearML.")
            task = None
    return task

def log_scalar(task, title: str, series: str, value: float, step: int):
    if task is None:
        return
    try:
        from clearml import Logger
        task.get_logger().report_scalar(title=title, series=series, value=value, iteration=step)
    except Exception as e:
        print(f"[ClearML] log_scalar error: {e}")

def log_figure(task, title: str, series: str, figure, step: int = 0):
    if task is None:
        return
    try:
        task.get_logger().report_matplotlib_figure(title=title, series=series, figure=figure, iteration=step)
    except Exception as e:
        print(f"[ClearML] log_figure error: {e}")

def upload_model(task, local_path: str, name: str = "best.pt"):
    if task is None:
        return
    try:
        task.update_output_model(model_path=local_path, name=name, auto_delete_file=False)
    except Exception as e:
        print(f"[ClearML] upload_model error: {e}")
