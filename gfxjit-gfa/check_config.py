#!/usr/bin/env python3
"""Standalone replica of TensileLite's validateCustomKernelMetadata checks.

Tensile.ValidateMetadata cannot run outside a built tree (it imports rocisa,
which is a build artifact), so this applies the same field requirements to the
.s directly. The build runs the real validator again via
Toolchain.Assembly.validateCustomKernelMetadataAtBuild.
"""
import sys
import yaml


def extract(path):
    lines = open(path).read().splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == ".amdgpu_metadata")
        end = next(i for i, l in enumerate(lines) if l.strip() == ".end_amdgpu_metadata")
    except StopIteration:
        raise RuntimeError("no .amdgpu_metadata block")
    body = "\n".join(lines[start + 1:end])
    doc = yaml.safe_load(body)
    if "custom.config" not in doc:
        raise RuntimeError("no custom.config key in the metadata")
    return doc, doc["custom.config"]


def main():
    path = sys.argv[1]
    doc, config = extract(path)
    missing = []

    isp = config.get("InternalSupportParams")
    if not isinstance(isp, dict):
        missing.append("InternalSupportParams")
    elif "KernArgsVersion" not in isp:
        missing.append("InternalSupportParams.KernArgsVersion")

    is_external = "Source" in config
    if is_external:
        source = config.get("Source")
        if not isinstance(source, dict) or "Origin" not in source:
            missing.append("Source.Origin")
        if not isinstance(config.get("Features"), dict):
            missing.append("Features")
        for f in ["Version", "ProblemType", "CustomKernel", "MatrixInstruction"]:
            if f not in config:
                missing.append(f)

    ck = config.get("CustomKernel")
    if ck is not None:
        if not isinstance(ck, dict):
            missing.append("CustomKernel (mapping)")
        else:
            for f in ["args", "macrotile", "threads", "grid"]:
                if f not in ck:
                    missing.append("CustomKernel." + f)
            if not isinstance(ck.get("uniformSummationOrder", False), bool):
                missing.append("CustomKernel.uniformSummationOrder (true or false)")

    kind = "External" if is_external else "TensileLite"
    if missing:
        print("INVALID (%s kernel): missing %s" % (kind, ", ".join(missing)))
        return 1
    print("VALID (%s kernel)" % kind)
    print("  Origin        :", config["Source"]["Origin"])
    print("  args          :", [a["semantic"] for a in ck["args"]])
    print("  macrotile     :", ck["macrotile"])
    print("  swizzle A/B   :", config["ProblemType"].get("SwizzleTensorA"),
          config["ProblemType"].get("SwizzleTensorB"))
    print("  MXScaleFormat :", config.get("MXScaleFormat"))
    print("  AssertSize    :", config.get("AssertSizeEqual"))
    print("  kernel name   :", doc["amdhsa.kernels"][0][".name"])
    return 0


sys.exit(main())
