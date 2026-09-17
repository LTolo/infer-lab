"""Hardware profiles for the heterogeneous fleet model.

A real inference fleet is never homogeneous: you have several accelerator
generations, different memory capacities and different interconnects, all
serving the same traffic.  Placement is therefore an optimisation problem, and
you cannot optimise what you have not characterised.

The numbers below are *published vendor peak specifications* used as modelling
inputs.  They are upper bounds -- achieved performance is what the roofline
model and the benchmark harness are for.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    peak_flops_bf16: float      # FLOP/s, dense
    memory_bandwidth: float     # bytes/s
    memory_bytes: int           # usable HBM
    interconnect_bytes_s: float # intra-node link bandwidth (NVLink-class)
    network_bytes_s: float      # inter-node fabric (InfiniBand-class)
    relative_cost: float        # cost per device-hour, normalised to the baseline

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity (FLOP/byte) where a kernel stops being memory-bound.

        Below the ridge point you are limited by HBM bandwidth no matter how
        good your kernel is; above it, by the tensor cores.  Decode for a dense
        transformer sits *far* below the ridge point on every modern accelerator,
        which is the entire economic argument for batching, quantization,
        GQA and speculative decoding.
        """
        return self.peak_flops_bf16 / self.memory_bandwidth

    def to_dict(self) -> dict[str, float | str | int]:
        return {
            "name": self.name,
            "peak_tflops_bf16": round(self.peak_flops_bf16 / 1e12, 2),
            "memory_bandwidth_gbs": round(self.memory_bandwidth / 1e9, 1),
            "memory_gb": round(self.memory_bytes / 1e9, 1),
            "nvlink_gbs": round(self.interconnect_bytes_s / 1e9, 1),
            "network_gbs": round(self.network_bytes_s / 1e9, 1),
            "ridge_point_flops_per_byte": round(self.ridge_point, 1),
            "relative_cost": self.relative_cost,
        }


# Three generations with deliberately different bandwidth/FLOP balance, so the
# placement policy has a genuinely non-trivial decision to make.
PROFILES: dict[str, HardwareProfile] = {
    "accel-a": HardwareProfile(
        name="accel-a",
        peak_flops_bf16=312e12,
        memory_bandwidth=2.0e12,
        memory_bytes=80 * 1000**3,
        interconnect_bytes_s=600e9,
        network_bytes_s=25e9,
        relative_cost=1.0,
    ),
    "accel-b": HardwareProfile(
        name="accel-b",
        peak_flops_bf16=989e12,
        memory_bandwidth=3.35e12,
        memory_bytes=80 * 1000**3,
        interconnect_bytes_s=900e9,
        network_bytes_s=50e9,
        relative_cost=2.6,
    ),
    "accel-c": HardwareProfile(
        name="accel-c",
        peak_flops_bf16=1979e12,
        memory_bandwidth=8.0e12,
        memory_bytes=192 * 1000**3,
        interconnect_bytes_s=1800e9,
        network_bytes_s=100e9,
        relative_cost=5.4,
    ),
}


def get_profile(name: str) -> HardwareProfile:
    if name not in PROFILES:
        raise KeyError(f"unknown hardware profile {name!r}; known: {sorted(PROFILES)}")
    return PROFILES[name]
