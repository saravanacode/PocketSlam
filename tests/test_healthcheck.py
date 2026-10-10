import asyncio

from aiohttp.test_utils import TestServer

from pocketslam.config import Settings
from pocketslam.healthcheck import check, healthz_url
from pocketslam.server.app import create_app


def test_url_uses_loopback_for_wildcard_bind(settings: Settings) -> None:
    s = settings.model_copy(update={"host": "0.0.0.0", "port": 9001})  # noqa: S104 - test input
    assert healthz_url(s) == "http://127.0.0.1:9001/healthz"


async def test_check_against_running_server(settings: Settings) -> None:
    async with TestServer(create_app(settings)) as server:
        url = str(server.make_url("/healthz"))
        assert await asyncio.to_thread(check, url)


def test_check_fails_when_nothing_listens() -> None:
    assert not check("http://127.0.0.1:9/healthz")
