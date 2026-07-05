"""wafer-deploy — serving + drift monitoring around the fixed wafer-mixed model.

No training lives here. The model is a frozen, calibrated artifact reused from a
sibling wafer-mixed checkout; this package is the life *around* it: a FastAPI
inference service and an unsupervised-first drift-monitoring stack.
"""
