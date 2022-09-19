import base64
import os
import pytest

from selenium import webdriver


def to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


@pytest.mark.hookwrapper
def pytest_runtest_makereport(item, call):
    pytest_html = item.config.pluginmanager.getplugin("html")
    outcome = yield
    report = outcome.get_result()
    extra = getattr(report, "extra", [])

    if report.when == "teardown":
        tmpdir = item.funcargs["tmpdir"]
        encoded = to_base64(f"{tmpdir}/screenshot.png")
        html = f"<div class='image'><img src='data:image/png;base64,{encoded}'></div>"
        extra.append(pytest_html.extras.html(html))
        report.extra = extra


@pytest.fixture
def selenium(tmpdir):
    firefox_options = webdriver.FirefoxOptions()
    selenium = webdriver.Remote(
        command_executor='http://localhost:4444/wd/hub',
        options=firefox_options
    )
    yield selenium
    selenium.save_screenshot(f"{tmpdir}/screenshot.png")
    selenium.quit()
