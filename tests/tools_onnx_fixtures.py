"""Tiny synthetic ONNX graphs matching each tool's graph contract.

They let the tests drive the real pre/post-processing and ONNX Runtime code
paths without the real weights (hundreds of MB, not reachable from CI).
"""
from __future__ import annotations

from pathlib import Path


def _save(model, path: Path) -> Path:
    import onnx

    path.parent.mkdir(parents=True, exist_ok=True)
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def _opset():
    from onnx import helper
    return [helper.make_opsetid("", 17)]


def upscale_x4(path: Path) -> Path:
    """[1,3,H,W] -> nearest x4."""
    from onnx import TensorProto, helper, numpy_helper
    import numpy as np

    scales = numpy_helper.from_array(np.array([1, 1, 4, 4], dtype=np.float32), "scales")
    node = helper.make_node("Resize", ["input", "", "scales"], ["output"], mode="nearest")
    g = helper.make_graph([node], "up", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "h", "w"])],
                          [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "H", "W"])], [scales])
    return _save(helper.make_model(g, opset_imports=_opset()), path)


def saliency(path: Path, size: int, *, n_outputs: int = 1) -> Path:
    """[1,3,S,S] -> channel mean [1,1,S,S] (n_outputs copies)."""
    from onnx import TensorProto, helper

    nodes = [helper.make_node("ReduceMean", ["input"], ["o0"], axes=[1], keepdims=1)]
    outs = [helper.make_tensor_value_info("o0", TensorProto.FLOAT, [1, 1, size, size])]
    for i in range(1, n_outputs):
        nodes.append(helper.make_node("Identity", ["o0"], [f"o{i}"]))
        outs.append(helper.make_tensor_value_info(f"o{i}", TensorProto.FLOAT, [1, 1, size, size]))
    g = helper.make_graph(nodes, "sal", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, size, size])], outs)
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    return _save(model, path)


def lama(path: Path) -> Path:
    """image [1,3,512,512], mask [1,1,512,512] -> image*255 with masked area = 255 (white)."""
    from onnx import TensorProto, helper, numpy_helper
    import numpy as np

    c255 = numpy_helper.from_array(np.array(255.0, dtype=np.float32), "c255")
    one = numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one")
    nodes = [
        helper.make_node("Sub", ["one", "mask"], ["keep"]),
        helper.make_node("Mul", ["image", "keep"], ["kept"]),
        helper.make_node("Add", ["kept", "mask"], ["filled"]),
        helper.make_node("Mul", ["filled", "c255"], ["output"]),
    ]
    g = helper.make_graph(nodes, "lama", [
        helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 512, 512]),
        helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, 512, 512])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 512, 512])], [c255, one])
    return _save(helper.make_model(g, opset_imports=_opset()), path)


def sam_encoder(path: Path) -> Path:
    """[1,3,1024,1024] -> [1,256,64,64] (avg-pooled, channel-tiled)."""
    from onnx import TensorProto, helper, numpy_helper
    import numpy as np

    reps = numpy_helper.from_array(np.array([1, 86, 1, 1], dtype=np.int64), "reps")
    starts = numpy_helper.from_array(np.array([0], dtype=np.int64), "starts")
    ends = numpy_helper.from_array(np.array([256], dtype=np.int64), "ends")
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), "axes")
    nodes = [
        helper.make_node("AveragePool", ["x"], ["p"], kernel_shape=[16, 16], strides=[16, 16]),
        helper.make_node("Tile", ["p", "reps"], ["t"]),
        helper.make_node("Slice", ["t", "starts", "ends", "axes"], ["image_embeddings"]),
    ]
    g = helper.make_graph(nodes, "enc", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 1024, 1024])],
                          [helper.make_tensor_value_info("image_embeddings", TensorProto.FLOAT, [1, 256, 64, 64])],
                          [reps, starts, ends, axes])
    return _save(helper.make_model(g, opset_imports=_opset()), path)


def sam_decoder(path: Path) -> Path:
    """Standard SAM decoder I/O; mask = left half of the original image (logits +1/-1)."""
    from onnx import TensorProto, helper, numpy_helper
    import numpy as np

    T = TensorProto.FLOAT
    inits = [
        numpy_helper.from_array(np.array([1, 1], dtype=np.int64), "one_one"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "z"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "o"),
        numpy_helper.from_array(np.array(2.0, dtype=np.float32), "two"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), "onef"),
        numpy_helper.from_array(np.array([[0.9]], dtype=np.float32), "iou_c"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "ax0"),
    ]
    nodes = [
        helper.make_node("Cast", ["orig_im_size"], ["hw"], to=TensorProto.INT64),
        helper.make_node("Concat", ["one_one", "hw"], ["shape"], axis=0),
        # x coordinate ramp along W: Range(0, W) -> compare with W/2
        helper.make_node("Slice", ["hw", "o", "shape_end", "ax0"], ["w1"]),
        helper.make_node("Squeeze", ["w1"], ["w"]),
        helper.make_node("Cast", ["w"], ["wf"], to=T),
        helper.make_node("Range", ["zf", "wf", "onef"], ["ramp"]),
        helper.make_node("Div", ["wf", "two"], ["half"]),
        helper.make_node("Less", ["ramp", "half"], ["left"]),
        helper.make_node("Cast", ["left"], ["leftf"], to=T),
        helper.make_node("Mul", ["leftf", "two"], ["l2"]),
        helper.make_node("Sub", ["l2", "onef"], ["row"]),          # +1 left, -1 right
        helper.make_node("ConstantOfShape", ["shape"], ["ones"], value=helper.make_tensor("v", T, [1], [1.0])),
        helper.make_node("Mul", ["ones", "row"], ["masks"]),
        helper.make_node("Identity", ["iou_c"], ["iou_predictions"]),
        # consume the other inputs so ORT keeps them as graph inputs
        helper.make_node("ReduceSum", ["image_embeddings"], ["s1"], keepdims=0),
        helper.make_node("ReduceSum", ["point_coords"], ["s2"], keepdims=0),
        helper.make_node("ReduceSum", ["point_labels"], ["s3"], keepdims=0),
        helper.make_node("ReduceSum", ["mask_input"], ["s4"], keepdims=0),
        helper.make_node("ReduceSum", ["has_mask_input"], ["s5"], keepdims=0),
        helper.make_node("Sum", ["s1", "s2", "s3", "s4", "s5"], ["low_res_masks"]),
    ]
    inits += [numpy_helper.from_array(np.array([2], dtype=np.int64), "shape_end"),
              numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zf")]
    ins = [helper.make_tensor_value_info("image_embeddings", T, [1, 256, 64, 64]),
           helper.make_tensor_value_info("point_coords", T, [1, "n", 2]),
           helper.make_tensor_value_info("point_labels", T, [1, "n"]),
           helper.make_tensor_value_info("mask_input", T, [1, 1, 256, 256]),
           helper.make_tensor_value_info("has_mask_input", T, [1]),
           helper.make_tensor_value_info("orig_im_size", T, [2])]
    outs = [helper.make_tensor_value_info("masks", T, [1, 1, "h", "w"]),
            helper.make_tensor_value_info("iou_predictions", T, [1, 1]),
            helper.make_tensor_value_info("low_res_masks", T, [])]
    g = helper.make_graph(nodes, "dec", ins, outs, inits)
    return _save(helper.make_model(g, opset_imports=_opset()), path)


def identity_face(path: Path) -> Path:
    """GFPGAN contract [1,3,512,512] in [-1,1] -> identity."""
    from onnx import TensorProto, helper

    g = helper.make_graph([helper.make_node("Identity", ["input"], ["output"])], "gfp",
                          [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 512, 512])],
                          [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 512, 512])])
    return _save(helper.make_model(g, opset_imports=_opset()), path)


def rife_mean(path: Path) -> Path:
    """img0, img1 [1,3,H,W], timestep [1,1,H,W] -> lerp(img0, img1, t)."""
    from onnx import TensorProto, helper

    T = TensorProto.FLOAT
    nodes = [helper.make_node("Sub", ["img1", "img0"], ["d"]),
             helper.make_node("Mul", ["d", "timestep"], ["dt"]),
             helper.make_node("Add", ["img0", "dt"], ["output"])]
    g = helper.make_graph(nodes, "rife", [
        helper.make_tensor_value_info("img0", T, [1, 3, "h", "w"]),
        helper.make_tensor_value_info("img1", T, [1, 3, "h", "w"]),
        helper.make_tensor_value_info("timestep", T, [1, 1, "h", "w"])],
        [helper.make_tensor_value_info("output", T, [1, 3, "h", "w"])])
    return _save(helper.make_model(g, opset_imports=_opset()), path)


def yunet_static(path: Path) -> Path:
    """YuNet 2023mar contract with its FIXED [1,3,640,640] input: constant
    outputs, one face at stride-32 cell (row 1, col 2), centre (80, 48) in
    graph pixels, 32x32 box."""
    from onnx import TensorProto, helper, numpy_helper
    import numpy as np

    T = TensorProto.FLOAT
    n = (640 // 32) ** 2
    score = np.zeros((1, n, 1), np.float32)
    score[0, 22] = 1.0
    bbox = np.zeros((1, n, 4), np.float32)
    bbox[0, 22] = [0.5, 0.5, 0.0, 0.0]
    kps = np.zeros((1, n, 10), np.float32)
    kps[0, 22] = [0.5, 0.5] * 5
    consts = {"cls_32": score, "obj_32": score, "bbox_32": bbox, "kps_32": kps}
    nodes = [helper.make_node("Constant", [], [k], value=numpy_helper.from_array(v, k + "_v"))
             for k, v in consts.items()]
    g = helper.make_graph(nodes, "yunet", [helper.make_tensor_value_info("input", T, [1, 3, 640, 640])],
                          [helper.make_tensor_value_info(k, T, list(v.shape)) for k, v in consts.items()])
    return _save(helper.make_model(g, opset_imports=_opset()), path)
