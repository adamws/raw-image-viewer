var Module = {
    canvas: (function () {
        return document.getElementById("canvas");
    })(),
};
Module["doNotCaptureKeyboard"] = true;

Module.onRuntimeInitialized = async () => {
    Module.api = {
        createBuffer: Module.cwrap("create_buffer", "number", ["number", "number", "number"]),
        loadTexture: Module.cwrap("load_textures", [], []),
        getPngSize: Module.cwrap("get_png_size", "number", []),
        getPngData: Module.cwrap("get_png_data", "number", []),
    };
    document.getElementById("downloadBtn").disabled = true;
};

function convert() {
    let files = document.getElementById("file").files;
    if (files.length == 0) {
        alert("Please select a file");
        return;
    }

    let width = document.getElementById("width").value;
    let height = document.getElementById("height").value;
    if (width <= 0 || height <= 0) {
        alert("Invalid file dimensions");
        return;
    }

    let file = files[0];
    let fr = new FileReader();
    fr.onload = function () {
        let format = parseInt(document.getElementById("format").value);
        let data = new Uint8Array(fr.result);
        const buffer = Module.api.createBuffer(width, height, format);
        Module.HEAPU8.set(data, buffer);
        Module.api.loadTexture();
        document.getElementById("canvas").removeAttribute("hidden");
        document.getElementById("downloadBtn").removeAttribute("disabled");
    };
    fr.readAsArrayBuffer(file);
}

function downloadAsPng() {
    let files = document.getElementById("file").files;
    if (files.length == 0) {
        alert("Please convert a file");
        return;
    }
    const pngData = Module.api.getPngData();
    const pngSize = Module.api.getPngSize();
    const view = new Uint8Array(Module.HEAPU8.buffer, pngData, pngSize);
    const blob = new Blob([view], {
        type: "octet/stream",
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = files[0].name + ".png";
    a.click();
    a.remove();
}

