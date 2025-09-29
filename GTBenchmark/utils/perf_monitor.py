# from __future__ import annotations
# """
# perf_monitor.py — GraphGym 集成 + 端到端(E2E)计时（OP Profile 可独占）
# --------------------------------------------------------------------
# - 直接使用 GraphGym 的 cfg（from GTBenchmark.graphgym.config import cfg）
# - 类变量一次性读 cfg.perf，避免反复访问
# - 当 enable_op_profiler=True 且 exclusive_op_profiler=True 时，关闭：
#   * E2E (iteration/section)
#   * 模块计时（ModuleTimer hooks）
#   * TB scalars/text（用 _NullWriter 吞掉）
# - 否则按常规：E2E + 模块计时 + TB 输出
# """
# import os, time, random, contextlib
# from contextlib import contextmanager
# from typing import Dict, Optional, Callable, List

# import torch
# from torch import nn
# from torch.utils.tensorboard import SummaryWriter

# # ---- GraphGym cfg：你明确说“一定有默认”，直接导入 ----
# from GTBenchmark.graphgym.config import cfg as CFG

# # ---- torch.profiler（保持兼容）----
# try:
#     from torch.profiler import profile, schedule, ProfilerActivity, ProfilerAction, tensorboard_trace_handler
#     _HAS_PROFILER = True
# except Exception:
#     profile = schedule = ProfilerActivity = ProfilerAction = tensorboard_trace_handler = None
#     _HAS_PROFILER = False

# # ---- DDP 默认通信 Hook（可选）----
# try:
#     from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook as _default_allreduce_hook
#     _HAS_DDP_HOOK = True
# except Exception:
#     _default_allreduce_hook = None
#     _HAS_DDP_HOOK = False


# # =========================
# #  小工具
# # =========================
# class _NullWriter:
#     def add_scalar(self, *a, **k): pass
#     def add_text(self, *a, **k): pass
#     def close(self): pass
#     def flush(self): pass


# class _Welford:
#     __slots__ = ("n","mean","M2")
#     def __init__(self):
#         self.n=0; self.mean=0.0; self.M2=0.0
#     def add(self, x: float):
#         self.n += 1
#         d = x - self.mean
#         self.mean += d / self.n
#         self.M2 += d * (x - self.mean)
#     def stats(self):
#         if self.n < 2:
#             return self.mean, 0.0
#         var = self.M2 / (self.n - 1)
#         return self.mean, var**0.5


# # =========================
# #  模块计时（叶子模块 Hook）
# # =========================
# class ModuleTimer:
#     def __init__(self, model: nn.Module, cfgd: Dict):
#         # 仅使用传入的 dict（来自 PerfMonitor 类变量一次性构建）
#         self.cfgd = cfgd
#         self.follow_windows = bool(cfgd["module_timers_follow_windows"] and cfgd["random_sampling"])
#         self.active = not self.follow_windows
#         self.handles: List[torch.utils.hooks.RemovableHandle] = []
#         self.reset_window()

#         def is_included(name:str)->bool:
#             inc = self.cfgd.get("include_modules", None)
#             exc = self.cfgd.get("exclude_modules", None)
#             if inc is not None and not any(s in name for s in inc):
#                 return False
#             if exc is not None and any(s in name for s in exc):
#                 return False
#             return True

#         for name, module in model.named_modules():
#             if len(list(module.children()))>0:
#                 continue  # 只挂叶子
#             if not is_included(name):
#                 continue
#             self._register_hooks(name, module)

#     def set_active(self, flag: bool):
#         self.active = bool(flag)

#     def _register_hooks(self, name: str, module: nn.Module):
#         use_cuda = bool(self.cfgd["use_cuda_events"]) and torch.cuda.is_available()
#         state: Dict[str,float] = {}

#         if use_cuda:
#             start_ev = torch.cuda.Event(enable_timing=True)
#             end_ev   = torch.cuda.Event(enable_timing=True)
#             def pre_hook(mod, inp):
#                 if not self.active: return
#                 start_ev.record()
#             def post_hook(mod, inp, out):
#                 if not self.active: return
#                 end_ev.record(); end_ev.synchronize()
#                 ms = start_ev.elapsed_time(end_ev)
#                 self._add_time(name, ms/1000.0)
#         else:
#             def pre_hook(mod, inp):
#                 if not self.active: return
#                 state["t0"] = time.perf_counter()
#             def post_hook(mod, inp, out):
#                 if not self.active: return
#                 t1 = time.perf_counter(); t0 = state.get("t0", t1)
#                 self._add_time(name, t1 - t0)

#         self.handles.append(module.register_forward_pre_hook(pre_hook))
#         try:
#             self.handles.append(module.register_forward_hook(post_hook))
#         except TypeError:
#             self.handles.append(module.register_forward_hook(lambda m,i,o: post_hook(m,i,o)))

#     def _add_time(self, name: str, sec: float):
#         acc = self.window_stats.get(name)
#         if acc is None:
#             acc = _Welford(); self.window_stats[name]=acc
#         acc.add(sec)

#     def reset_window(self):
#         self.window_stats: Dict[str,_Welford] = {}

#     def close(self):
#         for h in self.handles:
#             try: h.remove()
#             except Exception: pass
#         self.handles.clear()

#     def log_window_to_tb(self, writer: SummaryWriter, global_step: int, tag: str="train"):
#         for name, acc in self.window_stats.items():
#             mean, _ = acc.stats()
#             if acc.n>0:
#                 writer.add_scalar(f"{tag}/block_time/avg_sec/{name}", mean, global_step)
#         rows = ["| Module | mean(s) | ±95% CI (s) | n |", "|:--|--:|--:|--:|"]
#         any_row=False
#         def mean_of(acc): return acc.stats()[0]
#         for name, acc in sorted(self.window_stats.items(), key=lambda kv: mean_of(kv[1]), reverse=True):
#             mean, std = acc.stats(); n = acc.n
#             if n==0: continue
#             ci95 = 1.96*std/(n**0.5) if n>=2 else 0.0
#             rows.append(f"| {name} | {mean:.6f} | {ci95:.6f} | {n} |"); any_row=True
#         if any_row:
#             writer.add_text(f"{tag}/block_time/summary_ci95", "\n".join(rows), global_step)


# # =========================
# #  PerfMonitor（支持 OP 独占；cfg 类变量一次取）
# # =========================
# class PerfMonitor:

#     def __init__(self, model: nn.Module):
#         # ====== 一次性读取 cfg.perf 到类变量（后续实例共享）======
#         P = CFG.perf
#         # 基本路径
#         self.LOGDIR = P.logdir

#         # OP profiler & 独占策略（若某字段没在默认里，你也说“都有默认”，这里按直接访问）
#         self.ENABLE_OP_PROFILER = P.enable_op_profiler
#         self.EXCLUSIVE_OP = bool(getattr(P, "exclusive_op_profiler", True))
#         self.DISABLE_MT_WHEN_OP = bool(getattr(P, "disable_module_timer_when_op_profiler", True))
#         self.DISABLE_E2E_WHEN_OP = bool(getattr(P, "disable_e2e_when_op_profiler", True))
#         self.DISABLE_TB_WHEN_OP  = bool(getattr(P, "disable_tb_scalars_when_op_profiler", True))

#         # 随机窗口
#         self.RANDOM_SAMPLING   = bool(P.random_sampling)
#         self.SAMPLE_PROB       = float(P.sample_prob)
#         self.RANDOM_WARMUP     = int(P.random_warmup)
#         self.RANDOM_ACTIVE     = int(P.random_active)
#         self.RANDOM_COOLDOWN   = int(P.random_cooldown)
#         self.START_AFTER_STEPS = int(P.start_after_steps)

#         # 模块计时
#         self.MODULE_TIMERS_FOLLOW_WINDOWS = bool(P.module_timers_follow_windows)
#         self.INCLUDE_MODULES = list(P.include_modules) if getattr(P,"include_modules", None) is not None else None
#         self.EXCLUDE_MODULES = list(P.exclude_modules) if getattr(P,"exclude_modules", None) is not None else None
#         self.SCALAR_EVERY_N_STEPS = int(P.scalar_every_n_steps)
#         self.TEXT_TOPK_OPS = int(P.text_topk_ops)
#         self.USE_CUDA_EVENTS = bool(P.use_cuda_events)

#         # profiler 细节
#         self.RECORD_SHAPES = bool(P.record_shapes)
#         self.PROFILE_MEMORY = bool(P.profile_memory)
#         self.WITH_STACK = bool(P.with_stack)
#         self.WITH_MODULES = bool(P.with_modules)

#         # E2E
#         self.ENABLE_E2E = bool(P.enable_e2e)
#         self.E2E_LOG_EVERY = int(P.e2e_log_every_n_steps)
#         self.ENABLE_DDP_COMM_TIMING = bool(P.enable_ddp_comm_timing)

#         # 是否进入 OP 独占模式
#         self.OP_EXCLUSIVE = bool(self.ENABLE_OP_PROFILER and self.EXCLUSIVE_OP)
#         os.makedirs(self.LOGDIR, exist_ok=True)

#         # TB 写入
#         tb_enabled = not (self.OP_EXCLUSIVE and self.DISABLE_TB_WHEN_OP)
#         self.writer = SummaryWriter(log_dir=self.LOGDIR) if tb_enabled else _NullWriter()

#         self.global_step = 0

#         # 模块计时（OP 独占时可禁用）
#         self.model_timer = None
#         if not (self.OP_EXCLUSIVE and self.DISABLE_MT_WHEN_OP):
#             cfgd = dict(
#                 module_timers_follow_windows=self.MODULE_TIMERS_FOLLOW_WINDOWS,
#                 random_sampling=self.RANDOM_SAMPLING,
#                 include_modules=self.INCLUDE_MODULES,
#                 exclude_modules=self.EXCLUDE_MODULES,
#                 use_cuda_events=self.USE_CUDA_EVENTS,
#             )
#             self.model_timer = ModuleTimer(model, cfgd)

#         # 窗口调度（也控制 ModuleTimer 的 active）
#         self._rng = random.Random(12345)
#         self._schedule_fn = self._make_random_schedule() if self.RANDOM_SAMPLING else None
#         self._in_active = not self.RANDOM_SAMPLING

#         # OP 级 profiler
#         self._profiler = None
#         if _HAS_PROFILER and self.ENABLE_OP_PROFILER:
#             if self.RANDOM_SAMPLING:
#                 def tb_schedule(step: int):
#                     act = self._schedule_fn(step)
#                     if isinstance(act, str):
#                         if act == "WARMUP": return ProfilerAction.WARMUP
#                         if act == "RECORD": return ProfilerAction.RECORD
#                         return ProfilerAction.NONE
#                     return act
#                 self._profiler = profile(
#                     activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else []),
#                     schedule=tb_schedule,
#                     on_trace_ready=tensorboard_trace_handler(self.LOGDIR),
#                     record_shapes=self.RECORD_SHAPES,
#                     profile_memory=self.PROFILE_MEMORY,
#                     with_stack=self.WITH_STACK,
#                     with_modules=self.WITH_MODULES,
#                 )
#             else:
#                 self._profiler = profile(
#                     activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else []),
#                     schedule=schedule(wait=10, warmup=5, active=2, repeat=1),
#                     on_trace_ready=tensorboard_trace_handler(self.LOGDIR),
#                     record_shapes=self.RECORD_SHAPES,
#                     profile_memory=self.PROFILE_MEMORY,
#                     with_stack=self.WITH_STACK,
#                     with_modules=self.WITH_MODULES,
#                 )
#             self._profiler.__enter__()
#         else:
#             print("[PerfMonitor] Operator profiling disabled (enable_op_profiler=False or profiler unavailable).")

#         # E2E（OP 独占时可禁用）
#         self._e2e_enabled = bool(self.ENABLE_E2E) and not (self.OP_EXCLUSIVE and self.DISABLE_E2E_WHEN_OP)
#         self._e2e_log_every = int(self.E2E_LOG_EVERY)
#         self._e2e = None

#     # ---------- 随机窗口调度 ----------
#     def _make_random_schedule(self) -> Callable[[int], "ProfilerAction|str"]:
#         state = {"phase":"idle","remain":0,"cooldown":0}

#         def set_active(flag: bool):
#             self._in_active = flag
#             if self.model_timer and self.model_timer.follow_windows:
#                 self.model_timer.set_active(flag)

#         def act_none():
#             set_active(False)
#             return ProfilerAction.NONE if _HAS_PROFILER else "NONE"
#         def act_warm():
#             set_active(False)
#             return ProfilerAction.WARMUP if _HAS_PROFILER else "WARMUP"
#         def act_rec():
#             set_active(True)
#             return ProfilerAction.RECORD if _HAS_PROFILER else "RECORD"

#         def sched(step:int):
#             if step < self.START_AFTER_STEPS:
#                 return act_none()

#             if state["phase"]=="warmup":
#                 if state["remain"]>0:
#                     state["remain"]-=1; return act_warm()
#                 state["phase"]="active"; state["remain"]=max(0,self.RANDOM_ACTIVE); return act_rec()

#             if state["phase"]=="active":
#                 if state["remain"]>0:
#                     state["remain"]-=1; return act_rec()
#                 state["phase"]="idle"; state["cooldown"]=max(0,self.RANDOM_COOLDOWN); return act_none()

#             if state["cooldown"]>0:
#                 state["cooldown"]-=1; return act_none()

#             if random.random() < self.SAMPLE_PROB:
#                 state["phase"]="warmup" if self.RANDOM_WARMUP>0 else "active"
#                 state["remain"]=max(0, self.RANDOM_WARMUP if self.RANDOM_WARMUP>0 else self.RANDOM_ACTIVE)
#                 return act_warm() if self.RANDOM_WARMUP>0 else act_rec()

#             return act_none()
#         return sched

#     # ---------- E2E ----------
#     @contextmanager
#     def iteration(self, tag: str="train", batch_size: Optional[int]=None):
#         if not self._e2e_enabled:
#             yield; return

#         self._e2e = {
#             "tag": tag,
#             "t0": time.perf_counter(),
#             "sections": {},
#             "batch_size": batch_size,
#         }
#         try:
#             yield
#         finally:
#             total = time.perf_counter() - self._e2e["t0"]
#             if self.global_step % self._e2e_log_every == 0:
#                 self.writer.add_scalar(f"{tag}/iter/total_sec", total, self.global_step)
#                 if batch_size is not None and total > 0:
#                     self.writer.add_scalar(f"{tag}/throughput/samples_per_sec", float(batch_size)/total, self.global_step)
#                 for name, sec in self._e2e["sections"].items():
#                     self.writer.add_scalar(f"{tag}/iter/{name}_sec", sec, self.global_step)
#             self._e2e = None

#     @contextmanager
#     def section(self, name: str):
#         if not self._e2e_enabled:
#             yield; return
#         t0 = time.perf_counter()
#         try:
#             yield
#         finally:
#             sec = time.perf_counter() - t0
#             d = self._e2e["sections"]
#             d[name] = d.get(name, 0.0) + sec

#     # ---------- DDP 通信计时 ----------
#     def attach_ddp_comm_timing(self, ddp_model: torch.nn.parallel.DistributedDataParallel):
#         if not self.ENABLE_DDP_COMM_TIMING:
#             print("[PerfMonitor] DDP comm timing disabled by cfg.perf.enable_ddp_comm_timing=False")
#             return
#         if not _HAS_DDP_HOOK or _default_allreduce_hook is None:
#             print("[PerfMonitor] DDP default allreduce_hook not available; skip ddp_comm timing.")
#             return

#         def timing_hook(state, bucket):
#             t0 = time.perf_counter()
#             fut = _default_allreduce_hook(state, bucket)

#             def _done(fut):
#                 dt = time.perf_counter() - t0
#                 if self._e2e is not None:
#                     self._e2e["sections"]["ddp_comm"] = self._e2e["sections"].get("ddp_comm", 0.0) + dt
#                 return fut.value()

#             return fut.then(_done)

#         ddp_model.register_comm_hook(None, timing_hook)

#     # ---------- 其它 ----------
#     def profiled_step(self):
#         return contextlib.nullcontext()

#     def step(self, tag: str="train"):
#         self.global_step += 1

#         if self._schedule_fn is not None:
#             self._schedule_fn(self.global_step)

#         if self.model_timer and not self.model_timer.follow_windows:
#             if self.global_step % self.SCALAR_EVERY_N_STEPS == 0:
#                 self.model_timer.log_window_to_tb(self.writer, self.global_step, tag=tag)
#                 self.model_timer.reset_window()

#         if self.model_timer and self.model_timer.follow_windows and self._in_active:
#             self.model_timer.log_window_to_tb(self.writer, self.global_step, tag=tag)
#             self.model_timer.reset_window()

#         if self._profiler is not None:
#             self._profiler.step()

#     def summarize_profiler(self, topk_ops: Optional[int]=None, tag: str="train"):
#         if self._profiler is None:
#             return
#         try:
#             p = self._profiler
#             if hasattr(p, "key_averages"):
#                 ka = p.key_averages(group_by_input_shape=self.RECORD_SHAPES)
#                 sort_key = lambda e: getattr(e,"self_cuda_time_total",0.0) or getattr(e,"self_cpu_time_total",0.0)
#                 evts = sorted(ka, key=sort_key, reverse=True)
#                 k = int(topk_ops or self.TEXT_TOPK_OPS)
#                 lines = ["| # | Op | Self CUDA ms | Self CPU ms | Calls |","|:-:|:---|------------:|-----------:|------:|"]
#                 for i, evt in enumerate(evts[:k],1):
#                     cuda_ms = getattr(evt,"self_cuda_time_total",0.0)/1000.0
#                     cpu_ms  = getattr(evt,"self_cpu_time_total",0.0)/1000.0
#                     lines.append(f"| {i} | {evt.key} | {cuda_ms:.3f} | {cpu_ms:.3f} | {evt.count} |")
#                 self.writer.add_text(f"{tag}/top_ops", "\n".join(lines), self.global_step)
#         except Exception as e:
#             print(f"[PerfMonitor] summarize_profiler failed: {e}")

#     def close(self):
#         if self.model_timer:
#             self.model_timer.close()
#         try:
#             self.writer.flush(); self.writer.close()
#         except Exception:
#             pass
#         if self._profiler is not None:
#             try:
#                 self._profiler.__exit__(None, None, None)
#             except Exception:
#                 pass
