import os
import pytest

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec


INPUTS_PATH = os.environ.get("PYTEST_INPUTS_PATH", "./inputs")
WEBAPP = os.environ.get("WEBAPP", "localhost")


def test_canvas_displays_image(selenium):
    selenium.get(f"http://{WEBAPP}:6931")

    # fill inputs
    fill_with = {
        "file": f"{INPUTS_PATH}/320x240-chessboard-gradient.raw",
        "width": "320",
        "height": "240",
    }
    for k, v in fill_with.items():
        e = selenium.find_element("xpath", f"//input[@id='{k}']")
        e.send_keys(v)

    wait = WebDriverWait(selenium, 10)
    convert_btn = wait.until(
        ec.element_to_be_clickable((By.XPATH, "//button[text()='Convert']"))
    )
    convert_btn.click()

    canvas = wait.until(ec.visibility_of_element_located((By.XPATH, "//canvas[@id='canvas']")))
    assert canvas

