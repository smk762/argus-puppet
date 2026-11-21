"""The suite's age fail-safe for real-person input: argus-curator's MiVOLO estimates over HTTP.

The pose library may be built from real photos, and the suite keeps any frame where a face or a body reads under 18
out of it (background people included). The estimate comes from curator's ``POST /faces?body=true`` (InsightFace
faces plus a body pass, MiVOLO ages for both), the same service argus-darkroom's age check calls.

* :class:`AgeEstimator` is the protocol: :meth:`~AgeEstimator.min_age` gives the lowest estimate over every face and
  body in one image, ``None`` when there is nobody to estimate, and raises :class:`AgeUnavailable` when the check
  cannot vouch for the image (service down, an error answer, a face or body without an estimate, a body pass that
  did not run). Callers **fail closed**: an image that raised, or read ``None``, is excluded.
* :class:`HttpAgeEstimator` posts the image bytes to the URL in the settings (``ARGUS_PUPPET_AGE_URL``, never a URL a
  caller supplied), through :func:`argus_cortex.net.fetch` with that one host allowed, a per-read timeout and an
  overall deadline, and no redirects for the POST. Nothing is written.

An estimate is a tool's reading of an image, never a stated age. Nothing is sent anywhere unless a caller asks for
the check and a URL is configured.
"""

from __future__ import annotations

import json
import math
from typing import Protocol
from urllib.parse import urlsplit

import structlog

log = structlog.get_logger(__name__)

__all__ = ["ADULT_AGE", "AgeEstimator", "AgeUnavailable", "HttpAgeEstimator", "min_age_from_payload"]

# Any face or body read under this keeps the frame out.
ADULT_AGE = 18.0
# Per-read timeout and whole-call deadline of one check (curator queues for its GPU lease).
TIMEOUT_S = 30.0
DEADLINE_S = 60.0
# Curator's answer is a short JSON document; anything larger is not one.
MAX_REPLY_BYTES = 1024 * 1024
_PEOPLE_CHECK_FAILED = frozenset({"detection_failed", "body_ages_failed"})


class AgeUnavailable(RuntimeError):
    """The age check cannot vouch for this image (callers exclude it)."""


class AgeEstimator(Protocol):
    def min_age(self, data: bytes) -> float | None:
        """The lowest age estimate over every face and body in the image, in years; ``None`` when there is no face
        or body to estimate. Raises :class:`AgeUnavailable`."""
        ...


def _years(value: object) -> float | None:
    """An estimate, or ``None``. A bool, NaN, infinity or a negative number is no estimate (NaN < 18 is False, so
    accepting one would let the rule silently not fire)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        years = float(value)
    except OverflowError:
        return None
    return years if math.isfinite(years) and years >= 0 else None


def min_age_from_payload(payload: object) -> float | None:
    """The lowest estimate in curator's ``POST /faces?body=true`` answer; ``None`` when it found nobody.

    Raises :class:`AgeUnavailable` for anything that cannot vouch for the image: no ``faces`` list, a source other
    than MiVOLO, a body pass that did not report ``people_check: ok`` (without it an empty ``people`` list is not
    "nobody found"), a face without an estimate, or a body not matched to a face without one. A body matched to a
    face (an int index into ``faces``) may lack its own estimate (the face's counts); when it has one, it counts too.
    A body without one whose ``face_index`` is not a valid index fails closed."""
    if not isinstance(payload, dict) or not isinstance(payload.get("faces"), list):
        raise AgeUnavailable("the age check returned no faces list")
    if payload.get("source", "mivolo") != "mivolo":
        raise AgeUnavailable(f"the age check's source was {payload.get('source')!r}, not mivolo")
    check = payload.get("people_check")
    if check in _PEOPLE_CHECK_FAILED:
        raise AgeUnavailable(f"the body check failed ({check}): {payload.get('people_error') or ''}".rstrip(": "))
    if check != "ok":
        raise AgeUnavailable(f"the body check did not run (people_check {check!r})")
    people = payload.get("people")
    if not isinstance(people, list):
        raise AgeUnavailable("the age check returned no people list")
    ages: list[float] = []
    for face in payload["faces"]:
        years = _years(face.get("age_estimate_years")) if isinstance(face, dict) else None
        if years is None:
            raise AgeUnavailable("the age check returned a face without an estimate")
        ages.append(years)
    for person in people:
        if not isinstance(person, dict):
            raise AgeUnavailable("the age check returned a malformed person")
        years = _years(person.get("age_estimate_years"))
        if years is None:
            index = person.get("face_index")
            if index is None:
                raise AgeUnavailable("the age check returned a person without an estimate")
            # bool is an int: True would pass for face 1. Out of range would vouch with another face's age.
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(payload["faces"]):
                raise AgeUnavailable(f"the age check matched a person to face {index!r}, which it did not return")
            continue  # matched to a face, whose estimate is already counted
        ages.append(years)
    return min(ages) if ages else None


class HttpAgeEstimator:
    """MiVOLO estimates from argus-curator's ``POST /faces?body=true`` at *base_url* (from settings only)."""

    def __init__(self, base_url: str, *, timeout: float = TIMEOUT_S, deadline: float = DEADLINE_S) -> None:
        parts = urlsplit(base_url.strip())
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"ARGUS_PUPPET_AGE_URL must be an http(s) URL, not {base_url!r}")
        self.url = base_url.strip().rstrip("/") + "/faces?body=true"
        # The configured service is usually on loopback or the LAN: allow exactly its host (still resolved and
        # pinned by argus_cortex.net); every other host stays under the public-address rule.
        port = parts.port or (443 if parts.scheme == "https" else 80)
        host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
        self._allow = (f"{host}:{port}",)
        self.timeout = timeout
        self.deadline = deadline

    def min_age(self, data: bytes) -> float | None:
        try:
            from argus_cortex.net import FetchFailed, UrlRefused, fetch
        except ImportError as exc:  # pragma: no cover - httpx comes with the cli and server extras
            raise AgeUnavailable(f"the age check needs httpx (argus-cortex[remote]): {exc}") from exc
        try:
            reply = fetch(
                self.url,
                method="POST",
                content=data,
                headers={"Content-Type": "application/octet-stream"},
                allow_hosts=self._allow,
                timeout=self.timeout,
                deadline=self.deadline,
                max_bytes=MAX_REPLY_BYTES,
                max_redirects=0,
            ).raise_for_status()
        except FetchFailed as exc:
            raise AgeUnavailable(f"the age check answered HTTP {exc.status}") from exc
        except (UrlRefused, OSError) as exc:
            raise AgeUnavailable(f"the age check could not be reached: {exc}") from exc
        if 300 <= reply.status < 400:
            raise AgeUnavailable(f"the age check answered a redirect (HTTP {reply.status}); it is not followed")
        try:
            payload = json.loads(reply.body)
        except ValueError as exc:
            raise AgeUnavailable("the age check's answer is not JSON") from exc
        return min_age_from_payload(payload)
