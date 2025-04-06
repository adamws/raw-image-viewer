import base64
import os
import pytest

from selenium import webdriver


def is_circleci():
    return "CI" in os.environ


@pytest.fixture(scope="session")
def selenium_data_path():
    if is_circleci():
        return os.getcwd() + "/data"
    else:
        return "/home/seluser/data"


@pytest.fixture(scope="session")
def website():
    if is_circleci():
        return "http://localhost:6931"
    else:
        return "http://webapp:6931"


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


@pytest.fixture(scope="session")
def selenium_session(selenium_data_path):
    options = webdriver.FirefoxOptions()
    options.set_preference("browser.download.folderList", 2)
    options.set_preference("browser.download.manager.showWhenStarting", False)
    options.set_preference("browser.download.useDownloadDir", True)
    options.set_preference("browser.download.dir", f"{selenium_data_path}/outputs")
    selenium = webdriver.Remote(
        command_executor='http://localhost:4444',
        options=options
    )
    yield selenium
    selenium.quit()


@pytest.fixture
def selenium(selenium_session, website, tmpdir):
    selenium_session.get(website)
    yield selenium_session
    selenium_session.save_screenshot(f"{tmpdir}/screenshot.png")
