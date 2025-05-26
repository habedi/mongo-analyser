from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Label


class CollectionStatsWidget(Vertical):
    DEFAULT_CSS = """
    CollectionStatsWidget {
        height: auto;
        border: round $primary-darken-1;
        padding: 1;
    }
    CollectionStatsWidget > Label { margin-bottom: 1; text-style: bold; }
    """

    def compose(self) -> ComposeResult:
        yield Label("Collection Statistics")
        # In a real scenario, this DataTable would be populated
        # by data passed to this widget.
        yield DataTable(id="internal_collection_stats_table")

    def on_mount(self):
        table = self.query_one(DataTable)
        table.add_columns("Name", "Docs", "Avg Size", "Total Size", "Storage Size")

    # Add methods to update the table, e.g., update_stats(self, stats_data: list)
