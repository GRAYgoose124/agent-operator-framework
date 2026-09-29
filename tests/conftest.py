"""Shared test fixtures for AOF tests."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aof.config import (
    AppConfig,
    DiscoveryConfig,
    MemoryConfig,
    ResearchConfig,
    ToolsConfig,
    load_config,
)
from aof.memory.store import MemoryStore
from aof.tools.registry import ToolRegistry


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_config(temp_dir: Path) -> AppConfig:
    """AppConfig with temp paths for memory and research queue."""
    config = load_config()
    mem_config = MemoryConfig(
        db_path=str(temp_dir / "memory.db"),
        notes_dir=str(temp_dir / "memory"),
        memory_strategy="search",
        memory_search_limit=5,
    )
    tools_config = ToolsConfig(
        scripts_dir=str(temp_dir / "tools"),
        crawlers_dir=str(temp_dir / "crawlers"),
    )
    research_config = ResearchConfig(queue_path=str(temp_dir / "research_queue.json"))
    discovery_config = DiscoveryConfig(queue_path=str(temp_dir / "discovery_queue.json"))
    return AppConfig(
        model=config.model,
        pool=config.pool,
        memory=mem_config,
        tools=tools_config,
        pipeline=config.pipeline,
        evaluator=config.evaluator,
        local_server=config.local_server,
        models=config.models,
        roles=config.roles,
        research=research_config,
        discovery=discovery_config,
    )


@pytest.fixture
async def temp_memory(temp_config: AppConfig):
    """MemoryStore initialized with temp config."""
    memory = MemoryStore(temp_config.memory)
    await memory.initialize()
    yield memory
    await memory.close()


@pytest.fixture
async def temp_tools(temp_config: AppConfig, temp_memory: MemoryStore):
    """ToolRegistry with all builtins (search, scraper, nlp, math, file, memory)."""
    from aof.tools.builtin import (
        crawler_builder,
        file_tools,
        html_tools,
        math_tools,
        memory_tools,
        nlp,
        scraper,
        search,
    )

    tools = ToolRegistry()
    search.register(tools)
    scraper.register(tools)
    html_tools.register(tools)
    nlp.register(tools)
    math_tools.register(tools)
    file_tools.register(tools)
    memory_tools.register(tools, temp_memory)
    crawler_builder.register(tools, temp_config.tools.crawlers_dir)
    tools.load_scripts(Path(temp_config.tools.scripts_dir))
    return tools


@pytest.fixture
def mock_chat_model():
    """FakeListChatModel with canned responses for testing."""
    from langchain_core.language_models import FakeListChatModel

    return FakeListChatModel(
        responses=[
            "I have completed the task. The result is 4.",
            "Here is the research summary.",
        ]
    )


@pytest.fixture
def research_queue_fixture(temp_config: AppConfig):
    """ResearchQueue with temp path."""
    from aof.research import ResearchQueue

    return ResearchQueue(temp_config.research.queue_path)


# Default HTML for mock HTTP responses
MOCK_HTML = (
    "<html><head><title>Test Page</title></head>"
    "<body><p>Test content for scraping.</p></body></html>"
)


@pytest.fixture
def mock_backend():
    """Reusable InferenceBackend mock with configurable complete/complete_json."""
    from aof.inference.backend import CompletionResult, InferenceBackend

    class MockBackend(InferenceBackend):
        def __init__(
            self,
            complete_text: str = "Here are key findings: Python is popular and widely used for automation and data science.",
            complete_json_result: dict | None = None,
        ):
            self._complete = AsyncMock(
                return_value=CompletionResult(
                    text=complete_text,
                    tokens_used=10,
                    finish_reason="stop",
                )
            )
            self._complete_json = AsyncMock(
                return_value=complete_json_result or {"steps": ["Search the web", "Synthesize findings"]}
            )

        def model_info(self):
            return {"n_ctx": 2048, "max_tokens": 512, "model_path": "mock"}

        async def complete(self, messages, **kwargs):
            return await self._complete(messages, **kwargs)

        async def complete_json(self, messages, schema=None):
            return await self._complete_json(messages, schema=schema)

        async def start(self):
            pass

        async def shutdown(self):
            pass

    return MockBackend()


@pytest.fixture
def mock_http_scrape():
    """Context manager that patches urllib.request.urlopen to return mock HTML."""
    from unittest.mock import patch

    def _make_mock_response():
        mock = MagicMock()
        mock.read.return_value = MOCK_HTML.encode("utf-8")
        mock.headers.get_content_charset.return_value = "utf-8"
        mock.__enter__.return_value = mock
        mock.__exit__.return_value = None
        return mock

    class MockHTTPScrape:
        def __enter__(self):
            self._patch = patch(
                "urllib.request.urlopen",
                return_value=_make_mock_response(),
            )
            self._patch.start()
            return self

        def __exit__(self, *args):
            self._patch.stop()

    return MockHTTPScrape()


@pytest.fixture
def mock_web_search():
    """Context manager that patches web search to return canned results."""
    from unittest.mock import patch

    class MockWebSearch:
        def __enter__(self):
            self._patch = patch(
                "aof.tools.builtin.search._sync_search",
                return_value=[
                    {"title": "Python", "url": "https://example.com", "snippet": "Python is a language."},
                ],
            )
            self._patch.start()
            return self

        def __exit__(self, *args):
            self._patch.stop()

    return MockWebSearch()


def run_cli(args: list[str], cwd: Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run aof CLI with given args. Returns CompletedProcess."""
    full_args = ["uv", "run", "aof"] + args
    return subprocess.run(
        full_args,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


# Builtin tool names for pipeline validation (matches registry when all builtins registered)
BUILTIN_TOOL_NAMES = frozenset({
    "web_search", "web_news", "web_scrape", "add_citation", "add_to_research_queue", "file_read",
    "text_entities", "text_summarize", "numpy_stats",
    "refine_memory", "search_memory", "search_citations",
    "crawl_site", "crawl_sitemap", "scrape_with_playwright", "follow_links", "extract_structured",
    "extract_links", "extract_meta", "parse_html_fragment",
    "create_crawler", "propose_crawler_update",
})


def assert_valid_pipeline_toml(path: Path) -> None:
    """Validate TOML structure and that referenced tools exist in builtin registry."""
    import tomllib

    with open(path, "rb") as f:
        data = tomllib.load(f)
    steps = data.get("steps", [])
    if not steps:
        return  # Skip non-pipeline configs (e.g. extended_research.toml)
    assert len(steps) > 0, "Pipeline must have at least one step"

    valid_tools = BUILTIN_TOOL_NAMES

    for step in steps:
        assert "name" in step or "system_prompt" in step
        for tool in step.get("tools", []):
            assert tool in valid_tools, f"Unknown tool '{tool}' in step {step.get('name', 'unknown')}"
