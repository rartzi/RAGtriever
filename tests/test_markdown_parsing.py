from cortexindex.utils import parse_wikilinks, parse_tags

def test_parse_wikilinks():
    s = "See [[Note A]] and [[Folder/Note B#Section]]."
    assert parse_wikilinks(s) == ["Note A", "Folder/Note B"]

def test_parse_tags():
    s = "Discuss #architecture and #meeting-notes today."
    assert "architecture" in parse_tags(s)
