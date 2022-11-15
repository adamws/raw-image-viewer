# raw-image-viewer

Raw images viewer and `PNG` converter with OpenCV and Webassembly.

## Build

Clone with submodules and navigate to project root:

```
git clone --recursive https://github.com/adamws/raw-image-viewer.git
cd raw-image-viewer
```

and compile with local [`emsdk`](https://github.com/emscripten-core/emsdk) toolchain:

```
emcmake cmake . -B build
cmake --build build
```

or with docker:

```
docker run --rm -v $(pwd):/src -u $(id -u):$(id -g) emscripten/emsdk:3.1.21 \
  /bin/bash -c "emcmake cmake . -B build && cmake --build build"
```

When compiled, run locally with `emrun`:

```
emrun build/web/index.html
```
