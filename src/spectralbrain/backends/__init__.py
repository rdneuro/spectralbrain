"""SpectralBrain compute backends — CPU (NumPy/SciPy) and GPU (CuPy/JAX)."""

from spectralbrain.backends.cpu import (  # noqa: F401
    NumpyBackend, SamplerConfig, PyMCSampler, NutpieSampler,
    get_bayesian_sampler,
    parallel_map, parallel_batch, batch_iterator,
    MemoryInfo, ram_status, gc_collect,
    estimate_array_memory, memory_guard, shrink_array,
)
from spectralbrain.backends.gpu import (  # noqa: F401
    CupyBackend, JaxBackend, NumPyroSampler,
    VRAMInfo, vram_status, vram_clear, vram_defrag,
    vram_gc, vram_guard, vram_monitor,
    get_gpu_backend, get_gpu_bayesian_sampler,
)
