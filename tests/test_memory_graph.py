from nolane_personal.consolidation import MemoryLink
from nolane_personal.engine import LivingEngine
from nolane_personal.events import LivingEvent
from nolane_personal.memory import MemoryRecord
from nolane_personal.store import LivingStore


def test_memory_graph_is_append_only_and_queryable(tmp_path):
    store = LivingStore(tmp_path / "living.db")
    engine = LivingEngine(store)
    parent = MemoryRecord(text="source")
    child = MemoryRecord(text="derived", kind="inference")
    event = LivingEvent(kind="rest_cycle", source="test")
    link = MemoryLink(
        parent_memory_id=parent.memory_id,
        child_memory_id=child.memory_id,
        relation="consolidated_into",
        source_event_id=event.event_id,
        created_at=event.at,
    )
    store.commit_transition(event, engine.state, [parent, child], [link])
    assert store.memory_by_ids([parent.memory_id, child.memory_id]).keys() == {
        parent.memory_id,
        child.memory_id,
    }
    assert store.consolidated_parent_ids() == {parent.memory_id}
    links = store.memory_links()
    assert links[0]["child_memory_id"] == child.memory_id
    store.close()
