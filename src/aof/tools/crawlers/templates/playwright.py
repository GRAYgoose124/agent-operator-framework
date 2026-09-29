"""Playwright crawler template: for JS-rendered pages."""
import scrapy


class PlaywrightCrawlerSpider(scrapy.Spider):
    name = "playwright_{site_slug}"
    allowed_domains = ["{domain}"]
    start_urls = ["{start_url}"]
    custom_settings = {
        "DOWNLOAD_HANDLERS": {
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        "CONCURRENT_REQUESTS": 1,
        "ROBOTSTXT_OBEY": True,
    }

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(
                url,
                meta={"playwright": True, "playwright_include_page": True},
            )

    def parse(self, response):
        yield {
            "url": response.url,
            "title": response.css("title::text").get("") or "",
            "content": " ".join(
                t.strip() for t in response.css("body *::text").getall() if t.strip()
            )[:3000],
        }
