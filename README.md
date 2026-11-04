# cellcanvas-core

[![License](https://img.shields.io/pypi/l/cellcanvas-core.svg?color=green)](https://github.com/cellcanvas/cellcanvas-core/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/cellcanvas-core.svg?color=green)](https://pypi.org/project/cellcanvas-core)
[![Python Version](https://img.shields.io/pypi/pyversions/cellcanvas-core.svg?color=green)](https://python.org)
[![CI](https://github.com/cellcanvas/cellcanvas-core/actions/workflows/ci.yml/badge.svg)](https://github.com/cellcanvas/cellcanvas-core/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/cellcanvas/cellcanvas-core/branch/main/graph/badge.svg)](https://codecov.io/gh/cellcanvas/cellcanvas-core)

Core components for CellCanvas

Requires Python 3.11+ and Zarr 3. New annotations use Zarr format 3.

## Array dimensions

- Images: **CZYX** (channel, depth, row, column); **BCZYX** with a batch axis.
- 2D images keep **Z=1**. Add a channel axis to legacy ZYX images.
- Labels: **ZYX**, shared across channels; zero is unannotated. Add one to
  classifier predictions before saving as labels.
- Features: **DZYX**, where D is feature width. Use `feature_axis=-1` for
  ZYXD inputs. Keep feature order consistent between training and prediction.
