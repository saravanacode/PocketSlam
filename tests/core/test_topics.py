from pocketslam.core import topics


def test_topics_are_unique_absolute_paths() -> None:
    names = [v for k, v in vars(topics).items() if k.startswith("TOPIC_")]
    assert len(names) == len(set(names))
    assert all(name.startswith("/") for name in names)
