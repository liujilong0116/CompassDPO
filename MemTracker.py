import torch, contextlib, time

class MemTracker:
    def __init__(self, enable=True, rank=0):
        self.enable = enable and torch.cuda.is_available()
        self.rank = rank

    def _now(self):
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated()
        resv  = torch.cuda.memory_reserved()
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_resv  = torch.cuda.max_memory_reserved()
        return alloc, resv, peak_alloc, peak_resv

    def reset_peaks(self):
        if not self.enable: return
        torch.cuda.reset_peak_memory_stats()

    @contextlib.contextmanager
    def region(self, name: str):
        """测一个代码片段的峰值（since reset）以及片段前后 allocated/reserved。"""
        if not self.enable:
            yield
            return
        self.reset_peaks()
        a0,r0,_,_ = self._now()
        t0 = time.time()
        yield
        dt = time.time() - t0
        a1,r1,pa,pr = self._now()
        if self.rank == 0:
            print(f"[MEM] {name:18s} | "
                  f"alloc {a0/1e6:7.1f} -> {a1/1e6:7.1f} MB | "
                  f"peak_alloc {pa/1e6:7.1f} MB | "
                  f"resv {r0/1e6:7.1f} -> {r1/1e6:7.1f} MB | "
                  f"peak_resv {pr/1e6:7.1f} MB | "
                  f"{dt*1000:.0f} ms")
