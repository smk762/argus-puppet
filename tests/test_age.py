"""The age fail-safe client: curator's answer parsed fail-closed, and the HTTP call kept on a short leash."""

from __future__ import annotations

import json

import pytest
from argus_cortex import net

from argus_puppet.age import AgeUnavailable, HttpAgeEstimator, min_age_from_payload


def answer(faces=(), people=(), check="ok", **extra) -> dict:
    return {"source": "mivolo", "faces": list(faces), "people_check": check, "people": list(people), **extra}


def face(age, **kw) -> dict:
    return {"age_estimate_years": age, "bbox": [0, 0, 10, 10], **kw}


def test_the_youngest_face_or_body_counts() -> None:
    assert min_age_from_payload(answer([face(30.0), face(41.5)])) == 30.0
    body_only = {"bbox": [0, 0, 5, 9], "face_index": None, "age_estimate_years": 15.0}
    assert min_age_from_payload(answer([face(30.0)], [body_only])) == 15.0  # a background body with no face
    matched = {"bbox": [0, 0, 5, 9], "face_index": 0, "age_estimate_years": 17.0}
    assert min_age_from_payload(answer([face(30.0)], [matched])) == 17.0  # a matched body's own estimate counts
    assert min_age_from_payload(answer([face(30.0)], [{**matched, "age_estimate_years": None}])) == 30.0
    assert min_age_from_payload(answer()) is None  # nobody found


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {"faces": "no"},
        answer(source="fairface"),
        answer(check="detection_failed", people_error="boom"),
        answer(check="body_ages_failed"),
        answer(check=None),  # the body pass did not run: an empty people list is not "nobody"
        {"source": "mivolo", "faces": [], "people_check": "ok"},  # no people list
        answer([face(None)]),
        answer([face(float("nan"))]),
        answer([face(True)]),
        answer([face(-3)]),
        answer([face(30.0)], [{"bbox": [0, 0, 1, 1], "face_index": None, "age_estimate_years": None}]),
        answer([face(30.0)], ["junk"]),
    ],
)
def test_anything_that_cannot_vouch_is_unavailable(payload) -> None:
    with pytest.raises(AgeUnavailable):
        min_age_from_payload(payload)


class FakeFetch:
    def __init__(self, reply: net.Reply | Exception) -> None:
        self.reply = reply
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url: str, **kw) -> net.Reply:
        self.calls.append((url, kw))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _reply(status: int, payload: object) -> net.Reply:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return net.Reply(status=status, headers={}, body=body, url="http://curator.internal:8100/faces?body=true")


def test_the_http_client_posts_the_bytes_to_the_configured_host_only(monkeypatch) -> None:
    fake = FakeFetch(_reply(200, answer([face(25.0)])))
    monkeypatch.setattr(net, "fetch", fake)
    est = HttpAgeEstimator("http://curator.internal:8100/")
    assert est.min_age(b"image-bytes") == 25.0
    url, kw = fake.calls[0]
    assert url == "http://curator.internal:8100/faces?body=true"
    assert kw["method"] == "POST" and kw["content"] == b"image-bytes"
    assert kw["allow_hosts"] == ("curator.internal:8100",)
    assert kw["timeout"] > 0 and kw["deadline"] > 0 and kw["max_redirects"] == 0 and kw["max_bytes"] > 0


@pytest.mark.parametrize(
    "reply",
    [
        _reply(503, {"detail": "MiVOLO missing"}),
        _reply(307, b""),
        _reply(200, b"<html>not json</html>"),
        _reply(200, {"faces": []}),
        net.UrlRefused("could not connect to curator.internal"),
        OSError("network down"),
    ],
)
def test_the_http_client_fails_closed(monkeypatch, reply) -> None:
    monkeypatch.setattr(net, "fetch", FakeFetch(reply))
    with pytest.raises(AgeUnavailable):
        HttpAgeEstimator("http://curator.internal:8100").min_age(b"x")


def test_the_url_must_be_http() -> None:
    for bad in ("", "file:///etc/passwd", "curator:8100"):
        with pytest.raises(ValueError):
            HttpAgeEstimator(bad)
    assert HttpAgeEstimator("https://[::1]/")._allow == ("[::1]:443",)
