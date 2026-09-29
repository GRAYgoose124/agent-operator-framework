"""Generic crawler template: crawl all links up to depth limit."""
import scrapy
from urllib.parse import urlparse


class GenericCrawlerSpider(scrapy.Spider):
    name = "generic_{site_slug}"
    allowed_domains = ["{domain}"]
    start_urls = ["{start_url}"]
    custom_settings = {
        "CONCURRENT_REQUESTS": 2,
        "DOWNLOAD_DELAY": 0.5,
        "DEPTH_LIMIT": 3,
        "ROBOTSTXT_OBEY": True,
    }

    def parse(self, response):
        yield {
            "url": response.url,
            "title": response.css("title::text").get("") or "",
            "content": " ".join(
                t.strip() for t in response.css("body *::text").getall() if t.strip()
            )[:3000],
        }
        for href in response.css("a::attr(href)").getall():
            if href and (href.startswith(("http://", "https://", "/"))):
                yield response.follow(href, self.parse, errback=self.errback)

    def errback(self, failure):
        pass
