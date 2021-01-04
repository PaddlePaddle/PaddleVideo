[简体中文](../zh-CN.md) | English

# Tools

This page include the usage of some useful tools in PaddleVideo

## Params

To get the params of a model.

```bash
python3.7 tools/summary.py -c configs/recognization/tsm/tsm.yaml
```

## FLOPS
to print FLOPs.

```python
python3.7 tools/summary.py -c configs/recognization/tsm/tsm.yaml --FLOPs
```

## Test the export model <sup>coming soon</sup>
