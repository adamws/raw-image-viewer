import base64
import os
import re
from pathlib import Path

import cv2
import numpy as np
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


def trigger_file_convert(path, width, height, pixel_format, selenium):
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


def get_canvas_encoded(selenium):
    return selenium.execute_script(
        'return document.getElementById("canvas").toDataURL("image/png");'
    )


@pytest.mark.parametrize("path,width,height,pixel_format", input_files)
def test_canvas_with_reference_images(
    path, width, height, pixel_format, selenium, extra
):
    selenium.get(f"http://{WEBAPP}:6931")
    trigger_file_convert(path, width, height, pixel_format, selenium)

    canvas_encoded = get_canvas_encoded(selenium)
    extra.append(extras.html(f"<div class='image'><img src='{canvas_encoded}'></div>"))

    # compare with expected png
    with open(Path(path).with_suffix(".png"), "rb") as f:
        canvas_base64 = re.search(r"base64,(.*)", canvas_encoded).group(1)
        assert canvas_base64 == base64.b64encode(f.read()).decode("utf-8")


@pytest.mark.parametrize(
    "width,height",
    [
        (10, 10),
        (100, 100),
        (10, 100),
        (100, 10),
        (320, 240),
        (240, 320),
    ],
)
def test_canvas_with_generated_grey_images(width, height, selenium, tmpdir, extra):
    selenium.get(f"http://{WEBAPP}:6931")

    window_size = selenium.get_window_size()
    # input dimensions need to fit in window, otherwise canvas scaling will kick in
    # and image comparison will fail:
    assert width < window_size["width"], f"Browser width too small"
    assert height < window_size["height"], f"Browser height too small"

    data = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)
    path = f"{tmpdir}/data.raw"
    data.tofile(path)

    trigger_file_convert(path, width, height, "GREY8", selenium)

    canvas_encoded = get_canvas_encoded(selenium)
    extra.append(extras.html(f"<div class='image'><img src='{canvas_encoded}'></div>"))

    canvas_base64 = re.search(r"base64,(.*)", canvas_encoded).group(1)
    canvas_png = base64.b64decode(canvas_base64)
    canvas_data = cv2.imdecode(
        np.frombuffer(canvas_png, np.uint8), cv2.IMREAD_GRAYSCALE
    )

    np.testing.assert_array_equal(canvas_data, data)
