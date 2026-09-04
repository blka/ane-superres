#!/usr/bin/env python3
"""Re-batch a fixed-shape CoreML mlpackage (mlProgram) for ANE batching.

Takes an existing mlpackage converted with convert.py (input (1,3,H,W)) and
produces a copy whose batch dimension is N. The graph must be batch-agnostic
(conv/prelu/pixel_shuffle ops are; anything baking intermediate shapes via
reshape is not — the validator will refuse to compile).

Usage:
  rebatch.py MODEL.mlpackage N OUT.mlpackage

The SR driver reads the batch size from the model description, so no flag is
needed at inference time.
"""
import shutil
import sys

import coremltools as ct
from coremltools.proto import Model_pb2


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    src, batch, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)

    # the spec lives inside the package; edit it in place so weight.bin survives
    p = f"{dst}/Data/com.apple.CoreML/model.mlmodel"
    spec = Model_pb2.Model()
    with open(p, "rb") as f:
        spec.ParseFromString(f.read())

    def patch_batch(tt):
        dims = list(tt.dimensions)
        if (len(dims) == 4 and dims[0].WhichOneof("dimension") == "constant"
                and dims[0].constant.size == 1):
            dims[0].constant.size = batch
            return 1
        return 0

    nfix = 0
    for io in ("input", "output"):
        for f in getattr(spec.description, io):
            ma = f.type.multiArrayType
            if ma.shape and ma.shape[0] == 1:
                ma.shape[0] = batch
                nfix += 1
    fn = spec.mlProgram.functions["main"]
    nfix += patch_batch(fn.inputs[0].type.tensorType)
    for key in fn.block_specializations:
        for op in fn.block_specializations[key].operations:
            for out in op.outputs:
                if out.type.WhichOneof("type") == "tensorType":
                    nfix += patch_batch(out.type.tensorType)
    with open(p, "wb") as f:
        f.write(spec.SerializeToString())

    # compile + smoke-test: one predict with random data
    m = ct.models.MLModel(dst, compute_units=ct.ComputeUnit.ALL)
    d = spec.description.input[0].type.multiArrayType.shape
    x = np_random(d)
    m.predict({spec.description.input[0].name: x})
    print(f"saved {dst} (batch {batch}, {nfix} dimensions patched, predict OK)")


def np_random(shape):
    import numpy as np
    return np.random.rand(*shape).astype(np.float32)


if __name__ == "__main__":
    main()