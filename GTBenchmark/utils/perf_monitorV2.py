from __future__ import annotations
"""
perf_monitor_v2.py — 轻量、干净、GraphGym 友好的训练计时/日志

| mode      | E2E计时 | 模块计时 | OP profiler | 典型用途           |
|-----------|---------|----------|-------------|--------------------|
| off       | ✗       | ✗        | ✗           | 纯训练、零开销      |
| light     | ✓       | ✗        | ✗           | 常开看迭代/吞吐     |
| modules   | ✓       | ✓        | ✗           | 找最慢模块（Top-K） |
| op-only   | ✗       | ✗        | ✓（独占）    | 火焰图/调用栈/显存  |
| full      | ✓       | ✓        | ✓           | 深度诊断（短开）    |

要点
- TensorBoard 写入器惰性；仅 rank0 写；保留最近 N 个 events
- E2E：Welford + EWMA；一张 e2e.csv 同时包含【逐步】与【epoch 汇总】
- 模块计时：叶子模块、include/exclude、Top-K（TB + CSV）
- OP profiler：使用 torch.profiler.schedule；在每个 iter 调 `profiler.step()`
- NVTX 可选：iteration / section 打标
"""

import os, time, math, csv, glob, re
from contextlib import contextmanager
from typing import Dict, Optional, Callable, List, Tuple, Iterable

from GTBenchmark.graphgym.config import cfg as _CFG

import torch
from torch import nn
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler
from torch.utils.tensorboard.writer import SummaryWriter as _TBWriter



# =========================
# off调用
# =========================
class _NullWriter:
    def add_scalar(self, *a, **k): pass
    def add_text(self, *a, **k): pass
    def add_histogram(self, *a, **k): pass
    def flush(self): pass
    def close(self): pass

class _EWMA:
    __slots__ = ("avg", "alpha", "inited")
    def __init__(self, alpha: float=0.2):
        self.alpha = float(alpha); self.avg = 0.0; self.inited = False
    def add(self, x: float):
        x = float(x)
        if not self.inited:
            self.avg = x; self.inited = True
        else:
            self.avg = self.alpha * x + (1.0 - self.alpha) * self.avg
        return self.avg

class _Welford:
    __slots__ = ("n", "mean", "M2")
    def __init__(self):
        self.n = 0; self.mean = 0.0; self.M2 = 0.0
    def add(self, x: float):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.M2 += d * (x - self.mean)
    def stats(self) -> Tuple[float, float]:
        if self.n < 2: return self.mean, 0.0
        var = self.M2 / (self.n - 1)
        return self.mean, math.sqrt(var)
    def ci95(self) -> float:
        if self.n < 2: return 0.0
        _, std = self.stats()
        return 1.96 * std / math.sqrt(self.n)

def _is_rank0() -> bool:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
    except Exception:
        pass
    r = os.environ.get('RANK') or os.environ.get('LOCAL_RANK')
    return (r is None) or (str(r) == '0')

def _cleanup_old_events(logdir: str, keep_latest: int):
    """保留 logdir 下最近 keep_latest 个 tfevents 文件（非递归）。"""
    if keep_latest is None or keep_latest <= 0: return
    paths = sorted(glob.glob(os.path.join(logdir, 'events.out.tfevents.*')),
                   key=os.path.getmtime, reverse=True)
    for p in paths[keep_latest:]:
        try: os.remove(p)
        except Exception: pass


# =========================
# 模块计时
# =========================
class ModuleTimer:
    """
    模块计时器（按名称匹配而非层级深度）。
    按照 include/exclude/regex 规则匹配模型名称，加上按照name严格匹配module类名，并在初始化时打印详细匹配信息。
    """
    def __init__(
        self,
        model: nn.Module,
        *,
        use_cuda_events: bool = False,
        include_name: Optional[Iterable[str]] = [],
        include_regex: Optional[Iterable[str]] = None,
        exclude_regex: Optional[Iterable[str]] = None,
        topk_per_window: int = 20,
        verbose: bool = True,
    ):
        self.use_cuda_events = bool(use_cuda_events and torch.cuda.is_available())
        self.include_re = [re.compile(p) for p in (include_regex or [])]
        self.exclude_re = [re.compile(p) for p in (exclude_regex or [])]
        self.include_name = include_name
        self.topk = int(max(1, topk_per_window))
        self.verbose = verbose

        self.window: Dict[str, _Welford] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._pending: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []  #挂起事件队列
        self._register_hooks_verbose(model)


    # ---------------- 名称匹配判断 ----------------
    def _match_reason(self, name: str,module: nn.Module) -> Tuple[bool, str]:
        """
        返回: (是否选中, 匹配原因字符串)
        """
        cls_name = module.__class__.__name__
        # 1. 排除规则优先
        for r in self.exclude_re:
            if r.search(name):
                return False, f"{r.pattern}"
        ##严格匹配类名
        for r in self.include_name:
            if r==cls_name:
                return True,r
        # 2. 包含规则匹配
        for r in self.include_re:
            if r.search(name):
                return True, f"{r.pattern}"

        # 3. 如果没有 include 规则，则默认不过滤（false）
        if not self.include_re and not self.include_name:
            return True, "default (no include_regex provided)"

        # 4. 未命中任何规则
        return False, "not matched"


    # ---------------- 注册 hook 并打印匹配表 ----------------
    def _register_hooks_verbose(self, model: nn.Module):
        if self.use_cuda_events:
            def mk_hooks(name: str):
                # 通过Event记录事件计时module（正向和反向）
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev   = torch.cuda.Event(enable_timing=True)
                # start_ev_b = torch.cuda.Event(enable_timing=True)
                # end_ev_b   = torch.cuda.Event(enable_timing=True)
                dev = torch.device(_CFG.device)

                def pre(_m, _i):
                    
                    stream = torch.cuda.current_stream(dev)
                    stream.record_event(start_ev)   # 只记录，不同步

                def post(_m, _i, _o):
                    stream = torch.cuda.current_stream(dev)
                    stream.record_event(end_ev)     # 只记录，不同步
                    # 把事件对暂存，等到 iteration 末尾统一读取 elapsed_time
                    self._pending.append((f"{name}_fwd", start_ev, end_ev))

                # # backward hooks（记录反向）
                # def pre_bwd(_m, grad_input, grad_output):
                    
                #     stream = torch.cuda.current_stream(dev)
                #     stream.record_event(start_ev_b) 

                # def post_bwd(_m, grad_input, grad_output):
                #     stream = torch.cuda.current_stream(dev)
                #     stream.record_event(end_ev_b)    
                #     self._pending.append((f"{name}_bwd", start_ev_b, end_ev_b))
                # return pre, post, pre_bwd, post_bwd
                return pre, post
        else:
            def mk_hooks(name: str):
                state = {}
                def pre(_m, _i): state['t0'] = time.perf_counter()
                def post(_m, _i, _o):
                    t1 = time.perf_counter(); t0 = state.get('t0', t1)
                    self._add(name, t1 - t0)
                return pre, post

        matched, skipped = [], []
        table_rows = []
        for n, m in model.named_modules():
            if not n:
                continue

            ok, reason = self._match_reason(n,m)

            if not ok:
                skipped.append(n)
                table_rows.append(f"[ ] {n:<60} | {m.__class__.__name__:<20} | {reason}")
                continue

            pre, post = mk_hooks(n)
            self._handles.append(m.register_forward_pre_hook(pre))
            self._handles.append(m.register_forward_hook(post))
            # self._handles.append(m.register_full_backward_hook(lambda m, gi, go: pre_bwd(m, gi, go)))
            # self._handles.append(m.register_full_backward_hook(lambda m, gi, go: post_bwd(m, gi, go)))
            matched.append(n)
            table_rows.append(f"[*] {n:<60} | {m.__class__.__name__:<20} | {reason}")

        if self.verbose:
            print(f"\n[PerfMonitor] 模块匹配结果 (共 {len(matched)} 计时 / {len(skipped)} 跳过)：")
            print("-" * 110)
            print("模块名称                                                     | 类型                 | 匹配规则")
            print("-" * 110)
            for row in table_rows:
                print(row)
            print("-" * 110)
            print(f"[PerfMonitor] 已注册 {len(matched)} 个模块计时 hook\n")

    # ---------------- 数据统计 ----------------
    def _add(self, name: str, sec: float):
        acc = self.window.get(name)
        if acc is None:
            acc = self.window[name] = _Welford()
        acc.add(float(sec))

    # ---------------- 写入 TensorBoard ----------------
    def dump_to_tb(self, writer, step: int, tag: str='train'):
        if not self.window:
            return
        items = [(n, acc.stats()[0], acc) for n, acc in self.window.items() if acc.n > 0]
        items.sort(key=lambda x: x[1], reverse=True)
        items = items[:self.topk]

        for n, mean, _acc in items:
            writer.add_scalar(f"{tag}/block_time/avg_sec/{n}", mean, step)

        rows = ["| # | Module | mean(s) | ±95% CI (s) | n |",
                "|:-:|:--|--:|--:|--:|"]
        for i, (n, mean, acc) in enumerate(items, 1):
            rows.append(f"| {i} | {n} | {mean:.6f} | {acc.ci95():.6f} | {acc.n} |")
        writer.add_text(f"{tag}/block_time/summary_ci95", "\n".join(rows), step)

        # writer.add_histogram(f"{tag}/block_time/hist_mean",
        #                      torch.tensor([m for _, m, _ in items]), step)

    def reset(self):
        self.window.clear()

    def close(self):
        for h in self._handles:
            try: h.remove()
            except Exception: pass
        self._handles.clear()




# =========================
# PerfMonitor
# =========================
class PerfMonitor:
    """
    - off: 全关
    - light: 仅 E2E
    - modules: E2E + 模块计时
    - op-only: 仅 OP（独占，使用 schedule；每步 profiler.step()）
    - full: E2E + 模块计时 + OP
    """
    def __init__(self, model: Optional[nn.Module]=None):
        P = getattr(_CFG, 'perf', {}) or {}

        def _g(name, default):
            return getattr(P, name, P.get(name, default)) if hasattr(P, name) else P.get(name, default)

        # ---- 配置（不新增 cfg 字段）
        self.mode = str(_g('mode', 'light')).lower()
        baselogdir: str = str(_g('logdir', './runs/perf'))
        timestamp = time.strftime("-%Y%m%d-%H%M%S")
        self.logdir = baselogdir+timestamp
        self.rank_zero_only: bool = bool(_g('rank_zero_only', True))

        self.tb_flush_secs: int = int(_g('tb_flush_secs', 120))
        self.tb_max_queue: int = int(_g('tb_max_queue', 4000))
        self.tb_suffix: str = str(_g('tb_filename_suffix', ''))
        self.keep_latest_event_files: int = int(_g('keep_latest_event_files', 2))

        self.scalar_every: int = int(_g('scalar_every_n_steps', 1)) #每50步

        # E2E
        self.e2e_every: int = int(_g('e2e_log_every_n_steps', 1))
        self.ewma_alpha: float = float(_g('ewma_alpha', 0.2))

        # 模块计时
        self.use_cuda_events: bool = bool(_g('use_cuda_events', True))
        self.include_name = _g('include_name',[])
        self.include_regex = _g('include_regex', [])
        self.exclude_regex = _g('exclude_regex', [])
        self.topk_modules: int = int(_g('module_topk_per_window', 20))
        self.enable_module_timer_cfg: bool = bool(_g('enable_module_timer', False))

        # OP profiler（由 mode 决定是否启用；使用 schedule）
        self.enable_op_profiler: bool = bool(_g('enable_op_profiler', False))
        self.op_exclusive: bool = bool(_g('op_exclusive', True))
        self.record_shapes: bool = bool(_g('record_shapes', False))
        self.profile_memory: bool = bool(_g('profile_memory', False))
        self.with_stack: bool = bool(_g('with_stack', False))
        self.with_modules: bool = bool(_g('with_modules', False))

        # CSV
        self.csv_every: int = int(_g('csv_every_n_steps', 1))
        self.csv_e2e: str = str(_g('csv_filename_e2e', 'e2e.csv'))
        self.csv_modules: str = str(_g('csv_filename_modules', 'modules.csv'))

        # NVTX
        self.emit_nvtx: bool = bool(_g('emit_nvtx', False)) and torch.cuda.is_available()

        # ---- rank 过滤 / mode->功能
        self._is_active_rank = (not self.rank_zero_only) or _is_rank0()

        _do_e2e = self._is_active_rank and (self.mode in {'light', 'modules', 'full'})
        _do_modules = self._is_active_rank and (self.mode in {'modules', 'full'})
        if self._is_active_rank and self.mode == 'light' and self.enable_module_timer_cfg:
            _do_modules = True
        _do_op = self._is_active_rank and ((self.mode in {'op-only', 'full'}) or self.enable_op_profiler)

        if self.mode == 'op-only' and self.op_exclusive:
            _do_e2e = False
            _do_modules = False

        self._do_e2e = _do_e2e
        self._do_modules = _do_modules
        self._do_op = _do_op

        # ---- 完全关闭
        self._disabled = (self.mode == 'off') or (not self._is_active_rank)
        if self._disabled:
            self._writer = _NullWriter()
            self._profiler = None
            self._module_timer = None
            self._e2e_state = None
            self.global_step = 0
            self._csv_e2e = None
            self._csv_modules = None
            self._csv_opened = False
            self._last_e2e_row = None
            # epoch 累计
            self._epoch_active = False
            self._epoch_t0 = 0.0
            self._epoch_iter_sum_total = 0.0
            self._epoch_iter_count = 0
            return

        # ---- 目录 & 先清理旧 events
        os.makedirs(self.logdir, exist_ok=True)
        _cleanup_old_events(self.logdir, self.keep_latest_event_files)

        # ---- TB 写入器惰性
        self._writer = None
        def _get_writer():
            if self._writer is None:
                self._writer = _TBWriter(
                    log_dir=self.logdir,
                    flush_secs=self.tb_flush_secs,
                    max_queue=self.tb_max_queue,
                    filename_suffix=self.tb_suffix,
                )
            return self._writer
        self._get_writer: Callable[[], _TBWriter | _NullWriter] = _get_writer

        # ---- E2E（逐步）
        self.global_step = 0
        self._e2e_state = None
        self._e2e_acc = {'total': _Welford(), 'throughput': _Welford()}
        self._e2e_ewma = {'total': _EWMA(self.ewma_alpha), 'throughput': _EWMA(self.ewma_alpha)}
        self._last_e2e_row = None
        self.epoch_time = []

        # ---- E2E（epoch 汇总，写入同一 e2e.csv）
        self._epoch_active = False
        self._epoch_t0 = 0.0
        self._epoch_iter_sum_total = 0.0
        self._epoch_iter_count = 0

        # ---- 模块计时
        self._module_timer = None
        if self._do_modules and model is not None:
            self._module_timer = ModuleTimer(
                model,
                use_cuda_events=self.use_cuda_events,
                include_name=self.include_name,
                include_regex=self.include_regex,
                exclude_regex=self.exclude_regex,
                topk_per_window=self.topk_modules,
            )

        # ---- OP profiler（schedule；在 step() 中推进）
        self._profiler = None
        if self._do_op:
            acts = [ProfilerActivity.CPU]
            if torch.cuda.is_available(): acts.append(ProfilerActivity.CUDA)
            #! Profiler settings
            prof_sched = schedule(wait=10, warmup=5, active=5, repeat=1)
            self._profiler = profile(
                activities=acts,
                schedule=prof_sched,
                on_trace_ready=tensorboard_trace_handler(self.logdir),
                record_shapes=self.record_shapes,
                profile_memory=self.profile_memory,
                with_stack=self.with_stack,
                with_modules=self.with_modules,
            )
            self._profiler.__enter__()
            print(f"[PerfMonitor] Torch profiler scheduled (mode={self.mode}). "
                    f"wait=10, warmup=5, active=10, repeat=1. Trace dir: {self.logdir}")

        # CSV：惰性打开
        self._csv_e2e = None
        self._csv_modules = None
        self._csv_opened = False

    # ---------------- CSV 工具 ----------------
    def _csv_open_once(self):
        if self._csv_opened:
            return
        os.makedirs(self.logdir, exist_ok=True)
        try:
            self._csv_e2e = open(os.path.join(self.logdir, self.csv_e2e), 'w', newline='')
            if self._csv_e2e.tell() == 0:
                # 一张表：逐步 + epoch 汇总
                csv.writer(self._csv_e2e).writerow(
                    ['step', 'tag', 'total_sec', 'throughput', 'ewma_total', 'ewma_throughput',
                     'epoch_total_sec', 'epoch_avg_iter_sec']
                )
        except Exception:
            self._csv_e2e = None
        try:
            self._csv_modules = open(os.path.join(self.logdir, self.csv_modules), 'w', newline='')
            if self._csv_modules.tell() == 0:
                csv.writer(self._csv_modules).writerow(
                    ['step', 'tag', 'module', 'mean_sec', 'ci95', 'n']
                )
        except Exception:
            self._csv_modules = None
        self._csv_opened = True

    # ---------------- E2E：epoch 汇总（合并迭代指标，不留空） ----------------
    def epoch_start(self, tag: str='train'):
        if self._disabled: return
        self._epoch_active = True
        self._epoch_t0 = time.perf_counter()
        self._epoch_iter_sum_total = 0.0
        self._epoch_iter_count = 0
        self._epoch_tag = tag

    def epoch_end(self, tag: Optional[str]=None):
        if self._disabled or not self._epoch_active: return
        ep_tag = tag or getattr(self, "_epoch_tag", "train")
        torch.cuda.synchronize()
        epoch_total = time.perf_counter() - self._epoch_t0
        iters = max(1, int(self._epoch_iter_count))
        avg_iter_sec = self._epoch_iter_sum_total / iters

        # TensorBoard
        # 注意epoch/下的iter时间是算出来的，不是计时出来的。已停用
        w = self._get_writer()
        w.add_scalar(f"{ep_tag}/epoch/total_sec", epoch_total, self.global_step)
        # w.add_scalar(f"{ep_tag}/epoch/avg_iter_sec", avg_iter_sec, self.global_step)
        self.epoch_time.append(epoch_total)
        # CSV（合并最近一次迭代指标）
        self._csv_open_once()
        if self._csv_e2e is not None and not self._csv_e2e.closed:
            iter_cols = ['','','','']
            if self._last_e2e_row is not None:
                # columns: [step, tag, total_sec, throughput, ewma_total, ewma_throughput, '', '']
                iter_cols = self._last_e2e_row[2:6]
            row = [
                self.global_step, ep_tag,
                *iter_cols,
                f"{epoch_total:.6f}", f"{avg_iter_sec:.6f}"
            ]
            csv.writer(self._csv_e2e).writerow(row)
            # 这行已经写了，连带e2e一起清空，避免 step() 再写一次
            self._last_e2e_row = None

        self._epoch_active = False

    # ---------------- E2E（逐步） ----------------
    @contextmanager
    def profiled_step(self):
        yield

    @contextmanager
    def iteration(self, tag: str='train', batch_size: Optional[int]=None):
        if not self._do_e2e:
            yield; return

        t0 = time.perf_counter()
        sec_map: Dict[str, float] = {}
        if getattr(self, 'emit_nvtx', False):
            try: torch.cuda.nvtx.range_push(f"iter:{tag}")
            except Exception: pass
        try:
            self._e2e_state = {'tag': tag, 'secs': sec_map, 'batch_size': batch_size}
            yield
        finally:
            if getattr(self, 'emit_nvtx', False):
                try: torch.cuda.nvtx.range_pop()
                except Exception: pass

            torch.cuda.synchronize()
            total = time.perf_counter() - t0

            # === GPU event section timing (no ModuleTimer required in e2e) ===
            pending = sec_map.pop("_pending_gpu_events", None)
            if pending:
                for name, s_ev, e_ev in pending:
                    ms = s_ev.elapsed_time(e_ev)  # 已 sync，安全
                    sec_map[name] = sec_map.get(name, 0.0) + ms / 1000.0


            # （删除 section pending event 结算部分）
            # === ModuleTimer 的 pending 事件（GPU 真实执行时间）===
            if (self._module_timer is not None
                and getattr(self._module_timer, "use_cuda_events", False)
                and hasattr(self._module_timer, "_pending")):
                for name, s_ev, e_ev in self._module_timer._pending:
                    ms = s_ev.elapsed_time(e_ev)   # 毫秒
                    self._module_timer._add(name, ms / 1000.0)
                self._module_timer._pending.clear()

            self._e2e_state = None

            # 累计（逐步）
            self._e2e_acc['total'].add(total)
            ew_total = self._e2e_ewma['total'].add(total)

            if batch_size and total > 0:
                thr = float(batch_size) / total
                self._e2e_acc['throughput'].add(thr)
                ew_thr = self._e2e_ewma['throughput'].add(thr)
            else:
                thr = 0.0; ew_thr = self._e2e_ewma['throughput'].avg

            # 累计到 epoch
            if self._epoch_active:
                self._epoch_iter_sum_total += total
                self._epoch_iter_count += 1

            w_step = self.global_step + 1
            # TB（节流）
            if(w_step % self.e2e_every == 0):
                w = self._get_writer()
                w.add_scalar(f"{tag}/iter/total_sec", total, w_step)
                w.add_scalar(f"{tag}/iter/ewma_total_sec", ew_total, w_step)
                if thr > 0:
                    w.add_scalar(f"{tag}/throughput/samples_per_sec", thr, w_step)
                    w.add_scalar(f"{tag}/throughput/ewma_samples_per_sec", ew_thr, w_step)
                for name, sec in sec_map.items():
                    w.add_scalar(f"{tag}/iter/{name}_sec", sec, w_step)
                # w.add_histogram(f"{tag}/iter/total_hist", torch.tensor([total]), w_step)

            # CSV（逐步）：缓存行；等 step() 节流落盘

            self._last_e2e_row = [w_step, tag,
                                    f"{total:.6f}", f"{thr:.3f}", f"{ew_total:.6f}", f"{ew_thr:.3f}",
                                    '', '']  # 末两列留给 epoch 汇总
    
    
    @contextmanager
    def section(self, name: str):
        """
        section 计时器：
        - CPU: causal wall time
        - GPU (forward/backward only): CUDA event time（延迟结算）
        """
        if not self._do_e2e:
            yield
            return

        t0 = time.perf_counter()

        # ---------- GPU event (optional) ----------
        use_gpu_event = (
            name in ('forward', 'backward')
            and torch.cuda.is_available()
        )

        if use_gpu_event:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev   = torch.cuda.Event(enable_timing=True)
            torch.cuda.current_stream().record_event(start_ev)
            # print("device yes")

        try:
            yield
        finally:
            # ---------- CPU causal time ----------
            dt = time.perf_counter() - t0
            if self._e2e_state is not None:
                secs = self._e2e_state["secs"]
                secs[name] = secs.get(name, 0.0) + dt

            # ---------- GPU device time (deferred) ----------
            if use_gpu_event:
                torch.cuda.current_stream().record_event(end_ev)
                pending = self._e2e_state["secs"].setdefault("_pending_gpu_events", [])
                pending.append((f"{name}_device", start_ev, end_ev))



    def record(self, name: str, sec: float):
        """
        直接向当前 iteration session 写入一个计时项（无 with）
        适用于 dataloader / 外部等待 / Python 开销 等
        """
        if self._disabled:
            return
        if self._e2e_state is None:
            return
        try:
            sec = float(sec)
        except Exception:
            return

        secs = self._e2e_state["secs"]
        secs[name] = secs.get(name, 0.0) + sec


    # ----------------（可选）DDP 通信计时 ----------------
    def attach_ddp_comm_timing(self, ddp_model: torch.nn.parallel.DistributedDataParallel):
        try:
            from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook as _ar
        except Exception:
            print('[PerfMonitor] ddp_comm hook not available.')
            return
        def timing_hook(state, bucket):
            t0 = time.perf_counter(); fut = _ar(state, bucket)
            def _done(fut):
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                if self._e2e_state is not None:
                    secs = self._e2e_state['secs']
                    secs['ddp_comm'] = secs.get('ddp_comm', 0.0) + dt
                return fut.value()
            return fut.then(_done)
        ddp_model.register_comm_hook(None, timing_hook)

    # ---------------- 每步收尾 ----------------
    def step(self, tag: str='train'):
        self.global_step += 1

        # 推进 OP profiler（有则每步都 step）
        if self._profiler is not None:
            try:
                self._profiler.step()
            except Exception as e:
                if _is_rank0():
                    print(f"[PerfMonitor] profiler.step() error: {e}. Disabling profiler.")
                try:
                    self._profiler.__exit__(None, None, None)
                except Exception:
                    pass
                self._profiler = None

        # 写 E2E CSV（逐步，节流）
        if  self._last_e2e_row is not None and (self.global_step % self.csv_every == 0):
            self._csv_open_once()
            if self._csv_e2e is not None and not self._csv_e2e.closed:
                csv.writer(self._csv_e2e).writerow(self._last_e2e_row)
            self._last_e2e_row = None

        # 模块计时窗口输出
        if self._module_timer is not None and (self.global_step % self.scalar_every == 0):
            w = self._get_writer()
            self._module_timer.dump_to_tb(w, self.global_step, tag=tag)

            if self._module_timer.window:
                self._csv_open_once()
                if self._csv_modules is not None and not self._csv_modules.closed:
                    items = [(n, acc.stats()[0], acc)
                             for n, acc in self._module_timer.window.items() if acc.n > 0]
                    items.sort(key=lambda x: x[1], reverse=True)
                    items = items[:self.topk_modules]
                    wr = csv.writer(self._csv_modules)
                    for n, mean, acc in items:
                        wr.writerow([self.global_step, tag, n, f"{mean:.6f}", f"{acc.ci95():.6f}", acc.n])
            self._module_timer.reset()

    # ---------------- 关闭 ----------------
    def close(self):
        # 关 profiler（若仍在）
        if self._profiler is not None:
            try: self._profiler.__exit__(None, None, None)
            except Exception: pass
            self._profiler = None

        # 关模块计时 hook
        if self._module_timer is not None:
            try: self._module_timer.close()
            except Exception: pass

        # 关 CSV
        for name in ('_csv_e2e', '_csv_modules'):
            f = getattr(self, name, None)
            if f is not None:
                try: f.flush(); f.close()
                except Exception: pass
                finally: setattr(self, name, None)
        # === TensorBoard 写入全局统计 ===
        if self._writer is not None:
            try:
                w = self._get_writer()
            except Exception: return

            if w is not None:
            # 全局平均 iteration 耗时
                if self._e2e_acc["total"].n > 0:
                    mean_t, std_t = self._e2e_acc["total"].stats()
                    ci_t = self._e2e_acc["total"].ci95()
                    w.add_scalar("global/iter/avg_total_sec", mean_t, self.global_step)
                    w.add_scalar("global/iter/std_total_sec", std_t, self.global_step)
                    w.add_scalar("global/iter/ci95_total_sec", ci_t, self.global_step)
                    print(f"avg_iter_sec={mean_t:.6f}")

                # 全局平均吞吐量
                if self._e2e_acc["throughput"].n > 0:
                    mean_thr, std_thr = self._e2e_acc["throughput"].stats()
                    ci_thr = self._e2e_acc["throughput"].ci95()
                    w.add_scalar("global/throughput/avg_samples_per_sec", mean_thr, self.global_step)
                    w.add_scalar("global/throughput/std_samples_per_sec", std_thr, self.global_step)
                    w.add_scalar("global/throughput/ci95_samples_per_sec", ci_thr, self.global_step)
                    print(f"avg_throughput={mean_thr:.6f}")

                # === 全局平均 epoch 时间 ===
                if len(self.epoch_time) > 0:
                    avg_epoch = sum(self.epoch_time) / len(self.epoch_time)
                    var_epoch = sum((x - avg_epoch) ** 2 for x in self.epoch_time) / len(self.epoch_time)
                    std_epoch = math.sqrt(var_epoch)
                    w.add_scalar("global/epoch/avg_total_sec", avg_epoch, self.global_step)
                    w.add_scalar("global/epoch/std_total_sec", std_epoch, self.global_step)
                    if len(self.epoch_time) > 1:
                        ci95_epoch = 1.96 * std_epoch / math.sqrt(len(self.epoch_time))
                        w.add_scalar("global/epoch/ci95_total_sec", ci95_epoch, self.global_step)
                    print(f"avg_epoch_sec={avg_epoch:.6f}")
                # === 刷新并关闭 TensorBoard ===
                try:
                    w.flush()
                except Exception:
                    pass

                if self._writer is not None and not isinstance(self._writer, _NullWriter):
                    try:
                        self._writer.flush()
                        self._writer.close()
                    except Exception:
                        pass

        # 再清一次旧 events
        # try: _cleanup_old_events(self.logdir, self.keep_latest_event_files)
        # except Exception: pass
