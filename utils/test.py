import time

from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
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

# progress.start()
#
# task_id = progress.add_task("cluster-1 step 0", total=1000)
#
# print(progress.tasks[0])
# print(progress.tasks[0])
#
# for i in range(10):
#     # print(i)
#     progress.update(task_id, advance=10, description=f"cluster-1 step {i}", refresh=False)
#     time.sleep(1)
