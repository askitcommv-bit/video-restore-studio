# dev/

Not needed to run the app.

- `test_*.py`: ad-hoc debugging scripts that send graphs to a running ComfyUI (127.0.0.1:8188).
  Run them from the project root as modules so `config` and the relative paths resolve, e.g.
  `python -m dev.test_flattener`.
- `workflows/`: the original LTX-2.5 TiledFusion ComfyUI workflow templates. The app builds its
  ComfyUI API graphs in code (`restore_engine.py`); these are for loading into ComfyUI by hand.
