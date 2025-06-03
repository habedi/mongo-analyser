from textual.widgets import Label, DataTable

from mongo_analyser.widgets.collection_stats_widget import CollectionStatsWidget


def test_compose_and_on_mount(monkeypatch):
    widget = CollectionStatsWidget()

    comp = list(widget.compose())
    assert isinstance(comp[0], Label)
    assert comp[0].renderable == "Collection Statistics"
    assert isinstance(comp[1], DataTable)

    class DummyTable:
        def __init__(self):
            self.columns = []

        def add_columns(self, *cols):
            self.columns.extend(cols)

    dummy_table = DummyTable()
    monkeypatch.setattr(widget, "query_one", lambda *_args, **_kwargs: dummy_table)

    widget.on_mount()
    assert dummy_table.columns == ["Name", "Docs", "Avg Size", "Total Size", "Storage Size"]
