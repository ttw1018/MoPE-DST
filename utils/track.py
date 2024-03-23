from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def get_track():
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}% [{task.completed}/{task.total}]",
        "•",
        TimeElapsedColumn(),
        "|",
        TimeRemainingColumn(),
    )
    return progress
