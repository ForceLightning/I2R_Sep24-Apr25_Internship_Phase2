"""Optical flow calculation methods for video data."""

from __future__ import annotations

# Standard Library
from collections import deque
from typing import Sequence

# Third-Party
from matplotlib import animation, colors
from matplotlib import pyplot as plt

# Scientific Libraries
import numpy as np

# Image Libraries
import cv2
import PIL.ImageSequence as ImageSequence
from cv2 import typing as cvt
from PIL import Image

# PyTorch
import torch
from torchvision.transforms.v2 import functional as v2f

# First party imports
from utils.types import INV_NORM_RGB_DEFAULT, InverseNormalize


def dense_optical_flow(video: Sequence[cvt.MatLike]) -> list[cvt.MatLike]:
    """Compute dense optical flow on the CPU (slow).

    Args:
        video: Video frames to calculate optical flow with. Must be greyscale.

    Return:
        Optical flow.

    Raises:
        AssertionError: Video is not greyscale.

    """
    video = deque(video)
    assert all(
        frame.ndim == 2 for frame in video
    ), f"Video with frame of input shape {video[0].shape} must have only 2 dims: (height, width)."
    frame_1 = video[0]
    frame_1 = frame_1.reshape(*frame_1.shape, 1)  # View as (H, W, 1)

    flows: list[cvt.MatLike] = []
    video.rotate(-1)
    for frame in video:
        next = frame.reshape(*frame.shape, 1)

        # NOTE: The type hints in CV2 seem not to allow NoneType values, while
        # the documentation clearly shows that they are supported. Thus, static
        # type checking may throw errors below.
        flow = cv2.calcOpticalFlowFarneback(  # pyright: ignore[reportCallIssue]
            frame_1,
            next,
            None,  # pyright: ignore[reportArgumentType] False positive
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        flows.append(flow)
        frame_1 = next

    return flows


def cuda_optical_flow(
    video: Sequence[cvt.MatLike], threshold: float | None = None
) -> tuple[list[cvt.MatLike], list[cvt.MatLike] | None]:
    """Compute dense optical flow with hardware acceleration on NVIDIA cards.

    This method computes optical flow between all frames i with i + 1 for up to
    sequence length n-1, then computes for frame n and frame 0.

    Args:
        video: Video frames to calculate optical flow with. Must be greyscale.
        threshold: Threshold to apply to the cost buffer. If set, the cost buffer
            will be returned as well.

    Returns:
        Optical flow, and cost buffer (if threshold is set)

    Raises:
        AssertionError: CUDA support is not enabled.
        AssertionError: Video is not greyscale.
        AssertionError: Video frames are not of the same shape.
        RuntimeError: Error in calculating optical flow (see stack trace).

    """
    # GUARD: Check CUDA availability.
    num_cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
    assert num_cuda_devices, (
        f"Number of enabled cuda devices found: {num_cuda_devices}, must be > 0"
        + "OpenCV-contrib-python must be built with CUDA support enabled."
    )

    # GUARD: Video frame shape and dtype.
    seq_len = len(video)
    assert all(
        frame.ndim == 2 and frame.shape == video[0].shape and frame.dtype == np.uint8
        for frame in video
    ), (
        f"Video with frame of input shape {video[0].shape} must have only 2 dims: "
        + "(height, width), be all of the same shape, and be of type `numpy.uint8`"
    )

    if isinstance(threshold, float):
        assert 0.0 <= threshold and threshold <= 1.0, (
            "If threshold is set, must be of value 0.0 <= x <= 1.0, but is "
            + f"{threshold:.2f} instead"
        )

    cv2.cuda.resetDevice()
    h, w, *_ = video[0].shape

    # Initialise NVIDIA Optical Flow (SDK 2)
    nvof = cv2.cuda.NvidiaOpticalFlow_2_0.create(
        (h, w),
        perfPreset=cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_PERF_LEVEL_SLOW,
        outputGridSize=cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_OUTPUT_VECTOR_GRID_SIZE_1,
        hintGridSize=cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_HINT_VECTOR_GRID_SIZE_1,
        enableTemporalHints=True,
        enableCostBuffer=threshold is not None,
    )

    # OPTIM: Preallocate GPU memory for the GpuMats.
    cu_frame_1 = cv2.cuda.GpuMat(h, w, cv2.CV_8UC1)
    cu_frame_2 = cv2.cuda.GpuMat(h, w, cv2.CV_8UC1)
    cu_flow_raw = cv2.cuda.GpuMat(h, w, cv2.CV_16SC2)
    cu_flow_float = cv2.cuda.GpuMat(h, w, cv2.CV_8UC1)

    # Track all GPU allocated GpuMats for destruction.
    all_cu_frames = [cu_frame_1, cu_frame_2, cu_flow_raw, cu_flow_float]

    # Optionals if threshold is set.
    cu_cost: cv2.cuda.GpuMat | None = None
    cu_flow_threshed: cv2.cuda.GpuMat | None = None
    cu_cost_threshed: cv2.cuda.GpuMat | None = None
    if threshold is not None:
        cu_cost = cv2.cuda.GpuMat(h, w, cv2.CV_8UC1)
        cu_flow_threshed = cv2.cuda.GpuMat(h, w, cv2.CV_8UC1)
        cu_cost_threshed = cv2.cuda.GpuMat(h, w, cv2.CV_8UC1)
        all_cu_frames += [cu_cost, cu_flow_threshed, cu_cost_threshed]

    # OPTIM: Possible to use a BufferPool here?
    cu_stream = cv2.cuda.Stream()

    res: list[cvt.MatLike] = []
    res_cost: list[cvt.MatLike] | None = None
    if threshold is not None:
        res_cost = []

    for i in range(seq_len):
        # Transfer frames to the GPU.
        cu_frame_1.upload(video[i])
        cu_frame_2.upload(video[(i + 1) % seq_len])

        try:
            # NOTE: The type hints in CV2 seem not to allow NoneType values, while
            # the documentation clearly shows that they are supported. Thus, static
            # type checking may throw errors below.

            # Calculate flow.
            flow, cu_cost = nvof.calc(
                cu_frame_1,
                cu_frame_2,
                cu_flow_raw,
                stream=cu_stream,
                hint=None,
                cost=cu_cost,
            )

            cu_flow_float = nvof.convertToFloat(flow, cu_flow_float)
            if threshold is not None and res_cost is not None:
                _, cu_cost_threshed = cv2.cuda.threshold(
                    cu_cost,
                    threshold * 255,
                    255.0,
                    cv2.THRESH_TOZERO,
                    dst=cu_cost_threshed,
                    stream=cu_stream,
                )

                cu_flow_threshed = cu_flow_float.copyTo(
                    dst=cu_flow_threshed, mask=cu_cost_threshed, stream=cu_stream
                )

                res.append(cu_flow_threshed.download().astype(np.float32))
                res_cost.append(cu_cost.download().astype(np.uint8))
            else:
                res.append(cu_flow_float.download().astype(np.float32))
        except Exception as e:
            _cleanup(nvof, *all_cu_frames)
            raise RuntimeError(f"Error at iteration {i}") from e

    cu_stream.waitForCompletion()

    # Ensure GPU memory is fully released
    _cleanup(nvof, *all_cu_frames)
    nvof.collectGarbage()

    return res, res_cost


def _cleanup(nvof: cv2.cuda.NvidiaHWOpticalFlow, *cu_frames: cv2.cuda.GpuMat):
    # NOTE: If a BufferPool is used, deallocation must be done in a LIFO order.
    for cu_frame in cu_frames:
        cu_frame.release()
    nvof.collectGarbage()


# DEBUG: Plots the optical flow map with the original video.
def _plot_animated_with_flow(
    imgs,
    flows,
    inv_norm: InverseNormalize = INV_NORM_RGB_DEFAULT,
    show_plot: bool = True,
    title: str | None = None,
):
    fig = plt.figure(figsize=(8, 4))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122, projection="polar")
    ax2.set_theta_zero_location("N")  # pyright: ignore[reportAttributeAccessIssue]

    ims = []
    if isinstance(imgs, Image.Image):
        for i, image in enumerate(ImageSequence.Iterator(imgs)):
            im = ax1.imshow(image, animated=True)
            if i == 0:
                ax1.imshow(image)
            ims.append([im])
    else:
        for i in range(len(imgs)):
            img = imgs[i]
            flow = flows[i]
            c, h, w = flow.shape
            if c == 2:
                flow = flows[i].permute(1, 2, 0).detach().cpu().numpy()
                hsv = np.zeros((h, w, 3), dtype=np.uint8)
                hsv[..., 1] = 255
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(  # pyright: ignore[reportCallIssue]
                    mag,
                    None,  # pyright: ignore[reportArgumentType]
                    0,
                    255,
                    cv2.NORM_MINMAX,
                )
                flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            else:
                flow = np.asarray(v2f.to_pil_image(flow))

            img = np.asarray(v2f.to_pil_image(inv_norm(img).clamp(0, 1)))  # (H, W, C)
            combined_img = cv2.addWeighted(img, 1.0, flow, 0.9, 0)

            im = ax1.imshow(combined_img, animated=True)
            if i == 0:
                ax1.imshow(combined_img)
            ims.append([im])
    ani = animation.ArtistAnimation(
        fig, ims, interval=100 / 3, blit=True, repeat_delay=0
    )
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Colorwheel
    x_val = np.arange(0, 2 * np.pi, 0.01)
    y_val = np.ones_like(x_val)

    colormap = plt.get_cmap("hsv")
    norm = colors.Normalize(0.0, 2 * np.pi)
    ax2.scatter(x_val, y_val, c=x_val, s=300, cmap=colormap, norm=norm, linewidths=0)
    ax2.set_yticks([])

    if title:
        fig.suptitle(title)
    else:
        fig.suptitle("Optical Flow")

    if show_plot:
        plt.show()
    return ani


# DEBUG: Runs the function above.
if __name__ == "__main__":
    CINE_PATH = "data/train_val/Cine/4_1_0000.nii.tiff"

    _, cine_list = cv2.imreadmulti(CINE_PATH, flags=cv2.IMREAD_GRAYSCALE)
    flow_list, cost_list = cuda_optical_flow(cine_list)

    combined_video = torch.empty((30, 512, 512), dtype=torch.uint8)
    for i in range(30):
        img = cine_list[i]
        img = cv2.resize(img, (512, 512))
        combined_video[i, :, :] = torch.as_tensor(img)
    combined_video = combined_video.view(30, 1, 512, 512)

    combined_flow = torch.empty((30, 512, 512, 2), dtype=torch.uint8)
    for i in range(30):
        img = flow_list[i]
        img = cv2.resize(img, (512, 512))
        combined_flow[i, :, :, :] = torch.as_tensor(img)
    combined_flow = combined_flow.permute(0, 3, 1, 2)

    _plot_animated_with_flow(combined_video, combined_flow)
