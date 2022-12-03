import base64
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from pytest_html import extras
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec


INPUTS_PATH = "./data/inputs"

input_files = [
    ("320x240-chessboard-gradient-gray.raw", 320, 240, "GREY8"),
    ("320x240-chessboard-gradient-uyvy.raw", 320, 240, "UYVY"),
]


def click_convert_button(selenium):
    convert_btn = selenium.find_element("xpath", f"//button[text()='Convert']")
    assert convert_btn.is_displayed()
    assert convert_btn.is_enabled()
    convert_btn.click()


def assert_alert(selenium, message):
    try:
        WebDriverWait(selenium, 5).until(ec.alert_is_present())
        alert = selenium.switch_to.alert
        assert alert.text == message
        alert.accept()
    except TimeoutException:
        assert False, "No alert found"


def test_convert_without_file_input(selenium):
    click_convert_button(selenium)
    assert_alert(selenium, "Please select a file")


@pytest.mark.parametrize(
    "dimensions",
    [
        [None, None],
        [None, 0],
        [0, None],
        [0, 0],
        [-10, 10],
        [10, -10],
        ["a", 10],
        [10, "a"],
    ],
)
def test_convert_with_invalid_file_dimensions(dimensions, selenium):
    for k, v in zip(["width", "height"], dimensions):
        e = selenium.find_element("xpath", f"//input[@id='{k}']")
        if v != None:
            e.send_keys(str(v))
    e = selenium.find_element("xpath", f"//input[@id='file']")
    e.send_keys(f"{INPUTS_PATH}/{input_files[0][0]}")
    click_convert_button(selenium)
    assert_alert(selenium, "Invalid file dimensions")


def assert_browser_size(selenium, width, height):
    window_size = selenium.get_window_size()
    # input dimensions need to fit in window, otherwise canvas scaling will kick in
    # and image comparison will fail:
    assert width < window_size["width"], "Browser width too small"
    assert height < window_size["height"], "Browser height too small"


def trigger_file_convert(path, width, height, pixel_format, selenium):

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

    click_convert_button(selenium)

    wait = WebDriverWait(selenium, 10)
    canvas = wait.until(
        ec.visibility_of_element_located((By.XPATH, "//canvas[@id='canvas']"))
    )
    assert canvas


def get_canvas_encoded(selenium):
    return selenium.execute_script(
        'return document.getElementById("canvas").toDataURL("image/png");'
    )


@pytest.mark.parametrize("filename,width,height,pixel_format", input_files)
def test_canvas_with_reference_images(
    filename, width, height, pixel_format, selenium, extra
):
    path = f"{INPUTS_PATH}/{filename}"
    trigger_file_convert(path, width, height, pixel_format, selenium)

    canvas_encoded = get_canvas_encoded(selenium)
    extra.append(extras.html(f"<div class='image'><img src='{canvas_encoded}'></div>"))

    # compare with expected png
    # note that path here is in relation to python process, not selenium
    path = Path(path).with_suffix(".png")
    with open(path, "rb") as f:
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
    assert_browser_size(selenium, width, height)

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


@pytest.mark.parametrize(
    "color,width,height",
    [
        ("blue", 100, 100),
        ("green", 100, 100),
        ("red", 100, 100),
    ],
)
def test_canvas_with_generated_uyvy_images(
    color, width, height, selenium, tmpdir, extra
):
    assert_browser_size(selenium, width, height)

    bgr_data = np.zeros((height, width, 3), dtype=np.uint8)
    if color == "blue":
        bgr_data[:, :, 0] = 255
    elif color == "green":
        bgr_data[:, :, 1] = 255
    elif color == "red":
        bgr_data[:, :, 2] = 255
    else:
        raise Exception("Unrecognized color")

    # convert BGR to UYVY (not directly supported by cvtColor, need to do it with intermediate step)
    data = cv2.cvtColor(bgr_data, cv2.COLOR_BGR2YUV)
    y0 = np.expand_dims(data[..., 0][::, ::2], axis=2)
    u = np.expand_dims(data[..., 1][::, ::2], axis=2)
    y1 = np.expand_dims(data[..., 0][::, 1::2], axis=2)
    v = np.expand_dims(data[..., 2][::, ::2], axis=2)
    data = np.concatenate((u, y0, v, y1), axis=2)
    data = data.reshape((height, width, 2))

    path = f"{tmpdir}/data.raw"
    data.tofile(path)

    trigger_file_convert(path, width, height, "UYVY", selenium)

    canvas_encoded = get_canvas_encoded(selenium)
    extra.append(extras.html(f"<div class='image'><img src='{canvas_encoded}'></div>"))

    canvas_base64 = re.search(r"base64,(.*)", canvas_encoded).group(1)
    canvas_png = base64.b64decode(canvas_base64)
    canvas_data = cv2.imdecode(np.frombuffer(canvas_png, np.uint8), cv2.IMREAD_COLOR)

    # converting from brg -> uyvy -> bgr looses information, compare with some margin,
    # we are hoping to catch bugs like swapped channels here and not trying to
    # validate exact pixel values.
    np.testing.assert_allclose(canvas_data, bgr_data, atol=25)


def run_command_in_container(container_id, command):
    process = subprocess.run(
        f"docker exec -u root {container_id} {command}", shell=True, capture_output=True
    )
    return process.returncode, process.stdout.decode()


@pytest.fixture
def download_dir():
    outputs_path = "./data/outputs"
    os.makedirs(outputs_path, exist_ok=True)
    # mode change needed for selenium to write downloads there:
    os.chmod(outputs_path, 0o777)

    yield outputs_path

    # cleanup requires workarounds...
    # if selenium is run inside container, then downloaded png file has different file owner
    # than process running tests. The workaround is to run rm in running container.
    container_details = subprocess.run(
        "docker container ls --all | grep selenium", shell=True, capture_output=True
    )
    if container_details.returncode == 0:
        container_id = container_details.stdout.decode().split(" ", 1)[0]
        run_command_in_container(container_id, "rm -rf /home/seluser/data/outputs")
    else:
        shutil.rmtree(outputs_path)


def test_download_png_button(download_dir, selenium, tmpdir):
    (filename, width, height, pixel_format) = input_files[0]
    path = f"{INPUTS_PATH}/{filename}"
    trigger_file_convert(path, width, height, pixel_format, selenium)

    download_btn = WebDriverWait(selenium, 60).until(
        ec.element_to_be_clickable((By.XPATH, "//button[@id='downloadBtn']")), 60
    )
    download_btn.click()
    time.sleep(2)

    # compare with expected png
    name = Path(filename).stem
    input_path = f"{INPUTS_PATH}/{name}.png"
    downloaded_path = f"{download_dir}/{name}.raw.png"
    input_image = cv2.imdecode(np.fromfile(input_path), cv2.IMREAD_COLOR)
    download_image = cv2.imdecode(np.fromfile(downloaded_path), cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(input_image, download_image)


def test_download_png_button_without_convert(selenium):
    button = selenium.find_element("xpath", f"//button[@id='downloadBtn']")
    assert button.is_displayed()
    assert not button.is_enabled()
