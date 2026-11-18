"""Lightweight append-only log of the world model's predicted joint_pos
trajectory for a FIXED reference episode, recorded at every training update
(not just every sparse milestone) - cheap because we're only storing 6-float
joint angles, not full images or model weights. Used by
generate_joint_pos_timelapse_video.py to render a smooth, continuous
progression video (one frame per training update, not per milestone).

File format: a 8-byte header (magic + batch_length as int32), then records
appended back to back, one per training update:
    int64              update   (Dreamer._update_count this update happened at)
    T * 6 * float32              predicted joint_pos for each of T timesteps
                                   (T = batch_length: 5 real context + imagined rest)

The header makes the file self-describing - batch_length is read FROM the
file, never assumed from the caller's config. This matters because
--configs debug overrides batch_length (20 instead of the usual 64); without
a header, reading a trace back with the wrong assumed batch_length silently
misaligns every record and produces garbage values instead of a clear error.
"""

import pathlib
import struct

import numpy as np

_MAGIC = b"JPTR"


class TraceWriter:
    def __init__(self, path, batch_length):
        self.batch_length = batch_length
        path = pathlib.Path(path)
        write_header = not path.exists() or path.stat().st_size == 0
        self._file = open(path, "ab")
        if write_header:
            self._file.write(_MAGIC)
            self._file.write(struct.pack("<i", batch_length))
            self._file.flush()
        else:
            with open(path, "rb") as f:
                magic, existing_batch_length = _read_header(f)
            if existing_batch_length != batch_length:
                raise ValueError(
                    f"{path} already contains a trace written with batch_length="
                    f"{existing_batch_length}, but the current config uses "
                    f"batch_length={batch_length} - can't safely append records of "
                    f"a different size. Use a fresh --logdir or a matching config."
                )

    def write(self, update: int, joint_pos_trajectory: np.ndarray):
        assert joint_pos_trajectory.shape == (self.batch_length, 6)
        self._file.write(struct.pack("<q", update))
        self._file.write(joint_pos_trajectory.astype(np.float32).tobytes())
        self._file.flush()  # durability: survive a crash between writes

    def close(self):
        self._file.close()


def _read_header(f):
    magic = f.read(4)
    if magic != _MAGIC:
        raise ValueError(f"not a joint_pos trace file (bad magic bytes: {magic!r})")
    batch_length = struct.unpack("<i", f.read(4))[0]
    return magic, batch_length


def read_trace(path):
    """Returns (batch_length, updates: (N,) int64, trajectories: (N, T, 6) float32)."""
    with open(path, "rb") as f:
        _, batch_length = _read_header(f)
        data = f.read()
    record_size = 8 + batch_length * 6 * 4
    n_records = len(data) // record_size
    updates = np.empty(n_records, dtype=np.int64)
    trajectories = np.empty((n_records, batch_length, 6), dtype=np.float32)
    for i in range(n_records):
        chunk = data[i * record_size : (i + 1) * record_size]
        updates[i] = struct.unpack("<q", chunk[:8])[0]
        trajectories[i] = np.frombuffer(chunk[8:], dtype=np.float32).reshape(batch_length, 6)
    return batch_length, updates, trajectories
