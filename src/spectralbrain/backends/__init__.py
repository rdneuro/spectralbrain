"""SpectralBrain compute backends — CPU (NumPy/SciPy) and GPU (CuPy/JAX)."""

from spectralbrain.backends.cpu import (  # noqa: F401
    MemoryInfo,
    NumpyBackend,
    NutpieSampler,
    PyMCSampler,
    SamplerConfig,
    batch_iterator,
    estimate_array_memory,
    gc_collect,
    get_bayesian_sampler,
    memory_guard,
    parallel_batch,
    parallel_map,
    ram_status,
    shrink_array,
)
from spectralbrain.backends.gpu import (  # noqa: F401
    BlackjaxSampler,
    CupyBackend,
    JaxBackend,
    NumPyroSampler,
    TorchBackend,
    VRAMInfo,
    get_gpu_backend,
    get_gpu_bayesian_sampler,
    vram_clear,
    vram_defrag,
    vram_gc,
    vram_guard,
    vram_monitor,
    vram_status,
)
