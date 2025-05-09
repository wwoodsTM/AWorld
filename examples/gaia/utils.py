import json
from pathlib import Path
from typing import Any, Dict

from loguru import logger
from tabulate import tabulate


def load_dataset_meta(path: str, split: str = "validation"):
    data_dir = Path(path) / split

    dataset = []
    with open(data_dir / "metadata.jsonl", "r", encoding="utf-8") as metaf:
        lines = metaf.readlines()
        for line in lines:
            data = json.loads(line)
            if data["task_id"] == "0-0-0-0-0":
                continue
            if data["file_name"]:
                data["file_name"] = data_dir / data["file_name"]
            dataset.append(data)
    return dataset


def add_file_path(task: Dict[str, Any], file_path: str = "./gaia_dataset"):
    if task["file_name"]:
        file_path = Path(f"{file_path}/2023/validation/") / task["file_name"]
        if file_path.suffix in [".pdf", ".docx", ".doc", ".txt"]:
            task["Question"] += f" Here are the necessary document files: {file_path}"

        elif file_path.suffix in [".jpg", ".jpeg", ".png"]:
            task["Question"] += f" Here are the necessary image files: {file_path}"

        elif file_path.suffix in [".xlsx", "xls", ".csv"]:
            task["Question"] += (
                f" Here are the necessary table files: {file_path}, for processing excel file,"
                " you can use the excel tool or write python code to process the file"
                " step-by-step and get the information."
            )
        elif file_path.suffix in [".py"]:
            task["Question"] += f" Here are the necessary python files: {file_path}"

        else:
            task["Question"] += f" Here are the necessary files: {file_path}"

    return task


def report_results(entries):
    # Initialize counters
    total_entries = len(entries)
    total_correct = 0

    # Initialize level statistics
    level_stats = {}

    # Process each entry
    for entry in entries:
        level = entry.get("level")
        is_correct = entry.get("is_correct", False)

        # Initialize level stats if not already present
        if level not in level_stats:
            level_stats[level] = {"total": 0, "correct": 0, "accuracy": 0}

        # Update counters
        level_stats[level]["total"] += 1
        if is_correct:
            total_correct += 1
            level_stats[level]["correct"] += 1

    # Calculate accuracy for each level
    for level, stats in level_stats.items():
        if stats["total"] > 0:
            stats["accuracy"] = (stats["correct"] / stats["total"]) * 100

    # Print overall statistics with colorful logging
    logger.info("Overall Statistics:")
    overall_accuracy = (total_correct / total_entries) * 100

    # Create overall statistics table
    overall_table = [
        ["Total Entries", total_entries],
        ["Total Correct", total_correct],
        ["Overall Accuracy", f"{overall_accuracy:.2f}%"],
    ]
    logger.success(tabulate(overall_table, tablefmt="grid"))
    logger.info("")

    # Create level statistics table
    logger.info("Statistics by Level:")
    level_table = []
    headers = ["Level", "Total Entries", "Correct Answers", "Accuracy"]

    for level in sorted(level_stats.keys()):
        stats = level_stats[level]
        level_table.append(
            [level, stats["total"], stats["correct"], f"{stats['accuracy']:.2f}%"]
        )

    logger.success(tabulate(level_table, headers=headers, tablefmt="grid"))
