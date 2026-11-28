"""POST /control and ``extract.control_image``: one image straight to its control PNG (pose, then render)."""

from __future__ import annotations

import dataclasses
import io

import pytest
from argus_cortex.limits import BadInput, TooLarge
from conftest import FakeEstimator, draw_image, person
from fastapi.testclient import TestClient
from PIL import Image

from argus_puppet.extract import NobodyFound, control_image, pose_image
from argus_puppet.models import RenderOptions
from argus_puppet.render import render_png
from argus_puppet.server.app import PEOPLE_HEADER, create_app

PNG = {"content-type": "image/png"}


def client(settings, fake: FakeEstimator | None = None, **changes) -> TestClient:
    return TestClient(create_app(dataclasses.replace(settings, **changes), estimator=fake or FakeEstimator()))


def size(png: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(png)).size


# ── the library ──────────────────────────────────────────────────────────────────────────────────────────────────


def test_control_image_is_pose_then_render(settings, fake) -> None:
    image = draw_image(64, 48)
    options = RenderOptions(width=128, height=96, face=False)
    png, result = control_image(image, options, settings=settings, estimator=fake, name="a.png")
    assert result == pose_image(image, settings=settings, estimator=fake, name="a.png")
    assert png == render_png(result, options) and size(png) == (128, 96)
    assert len(result.people) == 2 and result.image == "a.png"


def test_control_image_defaults_to_the_images_own_size(settings, fake) -> None:
    png, _ = control_image(draw_image(64, 48), settings=settings, estimator=fake)
    assert size(png) == (64, 48)


def test_control_image_refuses_an_image_with_nobody_to_draw(settings) -> None:
    with pytest.raises(NobodyFound, match="nobody was found"):
        control_image(draw_image(), settings=settings, estimator=FakeEstimator(people=[]))


def test_control_image_caps_the_canvas_and_passes_bad_images_through(settings, fake) -> None:
    with pytest.raises(TooLarge, match="a 64x48 canvas"):
        control_image(draw_image(64, 48), settings=settings, estimator=fake, max_pixels=1000)
    with pytest.raises(BadInput):
        control_image(b"not an image", settings=settings, estimator=fake)


# ── POST /control ────────────────────────────────────────────────────────────────────────────────────────────────


def test_the_route_answers_the_png_the_library_draws(settings, fake) -> None:
    image = draw_image(64, 48)
    reply = client(settings, fake).post("/control", content=image, headers=PNG)
    assert reply.status_code == 200 and reply.headers["content-type"] == "image/png"
    assert reply.headers[PEOPLE_HEADER] == "2"
    assert reply.content == control_image(image, settings=settings, estimator=fake)[0]


def test_the_route_takes_the_render_options_as_query_parameters(settings, fake) -> None:
    image = draw_image(64, 48)
    query = "width=200&height=120&fit=skeleton&draw=body,hands&primary_only=true&min_score=0.5&stroke_scale=2"
    reply = client(settings, fake).post(f"/control?{query}", content=image, headers=PNG)
    assert reply.status_code == 200 and reply.headers[PEOPLE_HEADER] == "1"
    options = RenderOptions(
        width=200, height=120, fit="skeleton", face=False, primary_only=True, min_score=0.5, stroke_scale=2.0
    )
    assert reply.content == control_image(image, options, settings=settings, estimator=fake)[0]
    one_side = client(settings, fake).post("/control?width=128&stroke_scale=auto", content=image, headers=PNG)
    assert size(one_side.content) == (128, 96)  # one side alone keeps the image's aspect


def test_the_route_takes_a_multipart_file_and_a_face_hint(settings) -> None:
    fake = FakeEstimator(people=[person(2, 2, 20, 40, 0.9), person(30, 2, 20, 40, 0.5)])
    c = client(settings, fake)
    form = c.post("/control?primary_only=true&face=32,4,10,10", files={"file": ("a.png", draw_image(), "image/png")})
    assert form.status_code == 200
    hinted = pose_image(draw_image(), settings=settings, estimator=fake, face_hint=(32, 4, 10, 10))
    assert hinted.primary == 1  # the hint picked the second person, and that is who was drawn
    assert form.content == render_png(hinted, RenderOptions(primary_only=True))


def test_the_route_refuses_an_image_with_nobody_in_it(settings) -> None:
    reply = client(settings, FakeEstimator(people=[])).post("/control", content=draw_image(), headers=PNG)
    assert reply.status_code == 422 and "nobody was found" in reply.json()["detail"]


@pytest.mark.parametrize(
    ("query", "status", "detail"),
    [
        ("draw=legs", 400, "draw must list"),
        ("draw=,", 400, "draw must list"),
        ("fit=bogus-fit", 400, "fit:"),
        ("stroke_scale=thick", 400, "stroke_scale must be a number"),
        ("stroke_scale=0", 400, "stroke_scale:"),
        ("min_score=7", 400, "min_score:"),
        ("face=1,2,3", 400, "face must be x,y,w,h"),
        ("name=..", 400, "name"),
        ("width=0", 422, ""),
        ("height=9000", 422, ""),
    ],
)
def test_the_route_refuses_bad_options_without_echoing_them(settings, fake, query, status, detail) -> None:
    reply = client(settings, fake).post(f"/control?{query}", content=draw_image(), headers=PNG)
    assert reply.status_code == status and detail in str(reply.json()["detail"])
    assert "bogus-fit" not in reply.text and "thick" not in reply.text
    assert fake.calls == 0  # refused before anything was posed


def test_the_route_caps_the_canvas_before_and_after_the_image_is_read(settings, fake) -> None:
    c = client(settings, fake, max_pixels=5000)
    given = c.post("/control?width=100&height=100", content=draw_image(), headers=PNG)
    assert given.status_code == 413 and "100x100 canvas" in given.json()["detail"] and fake.calls == 0
    assert c.post("/control?width=50&height=50", content=draw_image(), headers=PNG).status_code == 200
    scaled = c.post("/control?width=200", content=draw_image(64, 48), headers=PNG)  # 200x150, known once posed
    assert scaled.status_code == 413 and "200x150 canvas" in scaled.json()["detail"]


def test_the_route_maps_image_errors_like_pose(settings, fake) -> None:
    c = client(settings, fake, max_upload_bytes=4096)
    assert c.post("/control", content=b"", headers=PNG).status_code == 400
    assert c.post("/control", content=b"not an image", headers=PNG).status_code == 415
    assert c.post("/control", content=b"{}", headers={"content-type": "application/json"}).status_code == 415
    assert c.post("/control", content=b"x" * 5000, headers=PNG).status_code == 413
    assert c.post("/control", content=draw_image(), headers={**PNG, "X-Argus-Priority": "urgent"}).status_code == 400


def test_the_route_writes_nothing(settings, fake, tmp_path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    reply = client(settings, fake, data_root=root).post("/control?name=a.png", content=draw_image(), headers=PNG)
    assert reply.status_code == 200 and list(root.rglob("*")) == []
