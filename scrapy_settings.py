# Scrapy settings for topical_crawler project

BOT_NAME = 'topical_crawler'

SPIDER_MODULES = ['topical_crawler.webcrawler']
NEWSPIDER_MODULE = 'topical_crawler.webcrawler'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 2

# Configure a delay for requests for the same website (default: 0)
DOWNLOAD_DELAY = 0.5

# The download delay setting will honor only one of:
CONCURRENT_REQUESTS_PER_DOMAIN = 2

# Disable cookies (enabled by default)
COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
TELNETCONSOLE_ENABLED = False

# Override the default request headers:
DEFAULT_REQUEST_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'de,en;q=0.5',
}

# Enable or disable spider middlewares
SPIDER_MIDDLEWARES = {
    'topical_crawler.middlewares.TopicalCrawlerSpiderMiddleware': 543,
}

# Enable or disable downloader middlewares
DOWNLOADER_MIDDLEWARES = {
    'topical_crawler.middlewares.TopicalCrawlerDownloaderMiddleware': 543,
}

# Configure item pipelines
ITEM_PIPELINES = {
    'topical_crawler.pipelines.TopicalCrawlerPipeline': 300,
}

# Enable and configure the AutoThrottle extension (disabled by default)
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 0.5
AUTOTHROTTLE_MAX_DELAY = 3.0
AUTOTHROTTLE_TARGET_CONCURRENCY = 2.0
AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 3600
HTTPCACHE_DIR = 'httpcache'
HTTPCACHE_IGNORE_HTTP_CODES = [503, 504, 400, 403, 404]
HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'

# Set User-Agent
USER_AGENT = 'TopicalCrawler/1.0 (+http://example.com/bot)'

# Configure logging
LOG_LEVEL = 'INFO'

# Memory usage optimization
REACTOR_THREADPOOL_MAXSIZE = 10
MEMUSAGE_ENABLED = True
MEMUSAGE_LIMIT_MB = 512
MEMUSAGE_WARNING_MB = 256

# Retry configuration
RETRY_ENABLED = True
RETRY_TIMES = 2
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]

# Timeout settings
DOWNLOAD_TIMEOUT = 15