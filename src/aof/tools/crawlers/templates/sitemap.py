"""Sitemap crawler template: use SitemapSpider for domain."""
import scrapy
from scrapy.spiders import SitemapSpider


class SitemapCrawlerSpider(SitemapSpider):
    name = "sitemap_{site_slug}"
    allowed_domains = ["{domain}"]
    sitemap_urls = ["{sitemap_url}"]
    custom_settings = {
        "CONCURRENT_REQUESTS": 2,
        "DOWNLOAD_DELAY": 0.5,
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
