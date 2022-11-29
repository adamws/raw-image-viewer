import os
import pytest

from pytest_html import extras

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec


INPUTS_PATH = os.environ.get("PYTEST_INPUTS_PATH", "./inputs")
WEBAPP = os.environ.get("WEBAPP", "localhost")

input_files = [
    (f"{INPUTS_PATH}/320x240-chessboard-gradient-gray.raw", 320, 240, "GREY8"),
    (f"{INPUTS_PATH}/320x240-chessboard-gradient-uyvy.raw", 320, 240, "UYVY"),
]


def get_canvas_encoded(selenium):
    return selenium.execute_script(
        'return document.getElementById("canvas").toDataURL("image/png");'
    )


@pytest.mark.parametrize("path,width,height,pixel_format", input_files)
def test_canvas_displays_image(path, width, height, pixel_format, selenium, extra):
    selenium.get(f"http://{WEBAPP}:6931")

    # fill inputs
    fill_with = {
        "file": path,
        "width": str(width),
        "height": str(height),
    }
    for k, v in fill_with.items():
        e = selenium.find_element("xpath", f"//input[@id='{k}']")
        e.send_keys(v)

    # select pixel format
    e = selenium.find_element("xpath", f"//select[@id='format']")
    e.send_keys(pixel_format)

    wait = WebDriverWait(selenium, 10)
    convert_btn = wait.until(
        ec.element_to_be_clickable((By.XPATH, "//button[text()='Convert']"))
    )
    convert_btn.click()

    canvas = wait.until(
        ec.visibility_of_element_located((By.XPATH, "//canvas[@id='canvas']"))
    )
    assert canvas

    canvas_encoded = get_canvas_encoded(selenium)
    extra.append(extras.html(f"<div class='image'><img src='{canvas_encoded}'></div>"))
