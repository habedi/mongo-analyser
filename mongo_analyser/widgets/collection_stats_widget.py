from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Label


class CollectionStatsWidget(Vertical):
    def compose(self) -> ComposeResult:
        yield Label("Collection Statistics")
        yield DataTable(id="internal_collection_stats_table")

    def on_mount(self):
        table = self.query_one(DataTable)
        table.add_columns("Name", "Docs", "Avg Size", "Total Size", "Storage Size")
