## Research steps and web tooling

This document summarizes the recommended **research workflow** and how to pick
between the available web tools. Pipelines can reference this behaviour in
their `system_prompt` text.

- **Baseline sequence**
  - Start with **memory**: call `search_memory` to see what has already been
    researched about the topic.
  - Run **web search**:
    - Use `web_search` for general background, documentation, static pages.
    - Use `web_news` when the question is time-sensitive or about recent
      events (markets, launches, politics, etc.).
  - Fetch **full pages** for the most important URLs:
    - Use `web_scrape` on 1–3 key result URLs when HTML is mostly static.
    - If `web_scrape` returns empty/partial content or the page is clearly
      JavaScript-driven, escalate to `scrape_with_playwright`.
  - For **many pages on one site**:
    - Use `crawl_sitemap` when the site exposes a good `sitemap.xml`.
    - Use `crawl_site` from a representative seed URL when there is no
      sitemap or it is incomplete.
    - Use `follow_links` when you want to constrain crawling to URLs that
      match a particular pattern (e.g. `/blog/`, `/docs/`).
  - As you go, pass promising follow-ups to the research backlog with
    `add_to_research_queue`.

- **Citations and storage**
  - When using the SOTA research pipeline, call `add_citation(url, title,
    snippet)` for every important source you rely on.
  - Downstream steps synthesize findings into reports and store them in
    memory with appropriate tags and links.

- **Future orchestration (not yet implemented)**
  - The framework may eventually support explicit orchestration signals such
    as `request_step` / `request_create_tool` and Model Context Protocol
    (MCP) integration so agents can ask the host to add new pipeline steps or
    tools dynamically.
  - For now, pipelines should encode the desired behaviour directly in
    TOML `steps` (system prompts, `tools` lists, and goal templates).

