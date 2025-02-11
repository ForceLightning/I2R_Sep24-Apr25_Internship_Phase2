# -*- coding: utf8 -*-
"""A rich progress bar that estimates the total time remaining.

Taken from https://github.com/Lightning-AI/pytorch-lightning/issues/3009
"""
# Standard Library
import re
from datetime import datetime, timedelta
from typing import override

# Third-Party
from rich.progress import ProgressColumn
from rich.style import Style
from rich.text import Text

# PyTorch
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBar


class RemainingTimeColumn(ProgressColumn):
    """Show total remaining time in training."""

    max_refresh = 1.0

    @override
    def __init__(self, style: str | Style) -> None:
        self.style = style
        self.estimated_time_per_epoch = None
        self.start_time = datetime.now()
        super().__init__()

    @override
    def render(self, task) -> Text:
        if "Epoch" in task.description:
            # Fetch current epoch number from task description.
            m = re.search(r"Epoch (\d+)/(\d+)", task.description)
            if not m:
                return Text("")
            current_epoch, total_epoch = int(m.group(1)), int(m.group(2))

            elapsed = task.finished_time if task.finished else task.elapsed
            elapsed = elapsed if elapsed else 0.0
            remaining = task.time_remaining

            if remaining:
                time_per_epoch = elapsed + remaining
                if self.estimated_time_per_epoch is None:
                    self.estimated_time_per_epoch = time_per_epoch
                else:
                    # Smooth the time_per_epoch estimation.
                    self.estimated_time_per_epoch = (
                        0.99 * self.estimated_time_per_epoch + 0.01 * time_per_epoch
                    )

                full_elapsed = datetime.now() - self.start_time
                remaining_total = (
                    self.estimated_time_per_epoch * (total_epoch - current_epoch - 1)
                    + remaining
                )

                remaining_total_td = timedelta(seconds=int(remaining_total))
                total_estimated_td = timedelta(
                    seconds=int(
                        remaining_total + elapsed + full_elapsed.total_seconds()
                    )
                )

                return Text(
                    f"• {remaining_total_td} / {total_estimated_td}", style=self.style
                )

        return Text("")


class BetterProgressBar(RichProgressBar):
    """A progress bar that estimates the total time remaining."""

    @override
    def configure_columns(self, trainer) -> list:
        columns = super().configure_columns(trainer)
        columns.insert(4, RemainingTimeColumn(style=self.theme.time))
        return columns
