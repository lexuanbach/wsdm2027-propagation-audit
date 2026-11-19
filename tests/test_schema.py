from docless.data import Cascade, Event
from docless.data.loaders import SyntheticLoader, generate_synthetic


def test_cascade_counts():
    c = Cascade(
        story_id="s1",
        label=1,
        events=[
            Event(1, 10, 100, 0.0),
            Event(2, 10, 100, 1.0),
            Event(1, 11, -1, 2.0),
        ],
    )
    assert c.num_users() == 2
    assert c.num_urls() == 2
    assert c.num_media() == 1  # the -1 should not count


def test_synthetic_round_trip(tmp_path):
    cascades = generate_synthetic(n_stories=10, seed=0)
    assert len(cascades) == 10
    assert all(c.events for c in cascades)
    assert {c.label for c in cascades} <= {0, 1}

    loader = SyntheticLoader(tmp_path)
    loader.save(cascades)
    reloaded = loader.load()
    assert len(reloaded) == 10
    assert reloaded[0].story_id == cascades[0].story_id
    assert reloaded[0].label == cascades[0].label
    assert len(reloaded[0].events) == len(cascades[0].events)


def test_synthetic_seeds_reproducible():
    a = generate_synthetic(n_stories=20, seed=42)
    b = generate_synthetic(n_stories=20, seed=42)
    assert [c.label for c in a] == [c.label for c in b]
    assert [c.num_users() for c in a] == [c.num_users() for c in b]
