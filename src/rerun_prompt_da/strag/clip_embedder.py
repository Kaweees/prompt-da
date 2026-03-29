"""
CLIP embedder — 512-dim ViT-B/32 embeddings for images and text.

Two backends:
  - GrpcClipEmbedder: connects to the gRPC CLIP server running on kaweees-dgx-spark
    (same server as commahackopenpilot/infrastructure/clip_server/server.py)
  - LocalClipEmbedder: runs CLIP locally via HuggingFace transformers + torch

Auto-selects: tries gRPC first if CLIP_HOST is reachable, else falls back to local.
"""
from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class ClipEmbedder(Protocol):
    def embed_image(self, frame: np.ndarray) -> list[float]: ...
    def embed_text(self, text: str) -> list[float]: ...


# ---------------------------------------------------------------------------
# gRPC backend (uses server on kaweees-dgx-spark)
# ---------------------------------------------------------------------------

class GrpcClipEmbedder:
    """Calls the gRPC CLIP server from commahackopenpilot/infrastructure/clip_server."""

    def __init__(self, host: str, port: int = 50051):
        import grpc

        # Inline proto stubs (matches clip.proto from commahackopenpilot)
        channel = grpc.insecure_channel(f"{host}:{port}")
        self._channel = channel
        self._host = host
        self._port = port
        self._stub = None
        self._pb2 = None

    def _get_stub(self):
        if self._stub is not None:
            return self._stub
        try:
            # Try importing from commahackopenpilot if on path
            from selfdrive.agent.clip_server import clip_pb2_grpc, clip_pb2
            self._pb2 = clip_pb2
            self._stub = clip_pb2_grpc.ClipServiceStub(self._channel)
        except ImportError:
            # Generate stubs on the fly from inline proto
            self._stub, self._pb2 = _make_grpc_stubs(self._channel)
        return self._stub

    def embed_image(self, frame: np.ndarray) -> list[float]:
        import cv2
        stub = self._get_stub()
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        req = self._pb2.EmbedRequest(image_bytes=buf.tobytes())
        resp = stub.EmbedImage(req)
        vec = list(resp.embedding)
        return _l2_normalize(vec)

    def embed_text(self, text: str) -> list[float]:
        stub = self._get_stub()
        req = self._pb2.EmbedTextRequest(text=text)
        resp = stub.EmbedText(req)
        vec = list(resp.embedding)
        return _l2_normalize(vec)


def _make_grpc_stubs(channel):
    """Dynamically compile the CLIP proto and return (stub, pb2).
    Only needed if clip_pb2 is not importable from commahackopenpilot.
    """
    import grpc
    from grpc_tools import protoc
    import tempfile, os, importlib, sys

    PROTO = """
syntax = "proto3";
service ClipService {
  rpc EmbedImage(EmbedRequest) returns (EmbedResponse);
  rpc EmbedText(EmbedTextRequest) returns (EmbedResponse);
}
message EmbedRequest { bytes image_bytes = 1; }
message EmbedTextRequest { string text = 1; }
message EmbedResponse { repeated float embedding = 1; }
"""
    tmpdir = tempfile.mkdtemp()
    proto_path = os.path.join(tmpdir, "clip.proto")
    with open(proto_path, "w") as f:
        f.write(PROTO)

    protoc.main([
        "grpc_tools.protoc",
        f"--proto_path={tmpdir}",
        f"--python_out={tmpdir}",
        f"--grpc_python_out={tmpdir}",
        proto_path,
    ])

    sys.path.insert(0, tmpdir)
    pb2 = importlib.import_module("clip_pb2")
    pb2_grpc = importlib.import_module("clip_pb2_grpc")
    stub = pb2_grpc.ClipServiceStub(channel)
    return stub, pb2


# ---------------------------------------------------------------------------
# Local backend (HuggingFace CLIP)
# ---------------------------------------------------------------------------

class LocalClipEmbedder:
    """Local CLIP using HuggingFace transformers. Falls back to CPU if no CUDA."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPProcessor, CLIPModel
        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP model {model_name} on {self._device}")
        self._model = CLIPModel.from_pretrained(model_name).to(self._device)
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model.eval()

    def embed_image(self, frame: np.ndarray) -> list[float]:
        from PIL import Image
        import torch
        img = Image.fromarray(frame[..., ::-1])  # BGR→RGB
        inputs = self._processor(images=img, return_tensors="pt").to(self._device)
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        vec = feats[0].cpu().numpy().tolist()
        return _l2_normalize(vec)

    def embed_text(self, text: str) -> list[float]:
        import torch
        inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(self._device)
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        vec = feats[0].cpu().numpy().tolist()
        return _l2_normalize(vec)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_embedder(
    clip_host: str | None = None,
    clip_port: int = 50051,
    force_local: bool = False,
) -> ClipEmbedder:
    """Return the best available embedder.

    Tries gRPC if clip_host is provided and reachable, else LocalClipEmbedder.
    """
    if not force_local and clip_host:
        try:
            import grpc
            channel = grpc.insecure_channel(f"{clip_host}:{clip_port}")
            grpc.channel_ready_future(channel).result(timeout=3)
            logger.info(f"Using gRPC CLIP server at {clip_host}:{clip_port}")
            return GrpcClipEmbedder(host=clip_host, port=clip_port)
        except Exception as e:
            logger.warning(f"gRPC CLIP unavailable ({e}), falling back to local CLIP")

    logger.info("Using local CLIP embedder (transformers)")
    return LocalClipEmbedder()


def _l2_normalize(vec: list[float]) -> list[float]:
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr /= norm
    return arr.tolist()
