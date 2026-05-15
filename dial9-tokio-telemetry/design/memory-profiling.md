# Memory Profiling

## Overview

Add sampled allocation profiling to dial9. A `Dial9Allocator<A>` wraps any
`GlobalAlloc` (default `System`) and emits `AllocEvent` / `FreeEvent` events
into the trace on sampled allocations. Stacks are captured via the frame
pointer unwinder we already use for CPU profiling. The viewer and analysis
toolkit gain allocation flamegraphs, per-task allocation totals, and leak
detection.

The hot path on the allocating thread is **bare bones**: sample decision,
stack capture, push a fixed-size POD record into a process-global lock-free
queue. A dedicated **consolidator thread** drains the queue and turns each
record into the corresponding trace event. This keeps the allocator hook
allocator-quiet by construction and lets the consolidator use ordinary
`HashMap`/encoder machinery without re-entering the hook (§5–§7 below).

The design mirrors jemalloc's and Go's allocation profilers: **geometric
(Poisson) sampling** keyed on allocation size. The expected sampling rate
is 1 sample per N bytes allocated, regardless of object size distribution.
This gives unbiased size-weighted profiles at bounded overhead.

**Why not delegate to jemalloc's built-in profiling?** jemalloc's `prof`
feature is excellent but allocator-specific — it doesn't work with the
system allocator, mimalloc, or any other `GlobalAlloc`. Our wrapper
approach works with *any* allocator, integrates directly into the dial9
trace (same timeline, same viewer, same task attribution), and captures
stacks via our existing frame-pointer unwinder (which is already
installed for CPU profiling). The tradeoff: jemalloc's profiler has
zero-cost access to internal metadata (bin sizes, arena stats) that we
can't see from outside. For users who need that level of allocator
internals, jemalloc's `prof` remains the right tool.

## Goals

- Always-on in production, with sub-1% overhead at default sampling rates.
- Works with any `GlobalAlloc` (system, jemalloc, mimalloc, etc.) via a
  zero-cost wrapper.
- Stacks tied to the worker thread + task that performed the allocation, so
  allocation hot paths can be attributed to specific tasks or poll ranges.
- Optional live-set tracking for leak detection. Off by default — it
  adds an extra ring push per dealloc and a per-sample lookup on the
  consolidator side.
- No symbolication on the hot path. Addresses flow through the existing
  background `SymbolizeProcessor`.
- **Install-once, captured handle.** `MemoryProfiler::install(handle)`
  sets a process-global static that the allocator reads. The captured
  handle routes all sampled events through the central recorder.
- **Instruments all threads**, not just tokio workers — allocations on
  tokio workers, blocking pool, user OS threads, and library-spawned
  threads are all recorded identically.
- **Viewer and analysis toolkit support.** Allocation flamegraphs,
  per-task allocation totals, and leak detection views ship alongside
  the backend instrumentation.

## Non-goals

- Tracking every allocation. We sample.
- Replacing heaptrack / valgrind. Those record every allocation and are
  fine for dev-time analysis. Dial9 is for production, always-on.
- Detecting use-after-free, double-free, or other memory safety bugs.
  That's what Miri and ASAN are for.
- **Inspecting memory contents.** We record size, address, and stack —
  never the bytes stored in the allocation. No PII/secret leakage risk.
- **Cryptographic randomness.** The per-thread RNG is a fast PRNG
  (Xoshiro256++) seeded from a user-supplied or random seed. It is not
  cryptographically secure and doesn't need to be — sampling decisions
  are not security-sensitive.
- Tracking stack allocations or `mmap`-backed memory outside the
  allocator. A future `RssSample` event could sample `/proc/self/statm`,
  but that's out of scope here. **Evolution path:** once the core
  allocator profiling is stable, we can add `mmap`/`munmap` interception
  via `LD_PRELOAD` or a similar mechanism to cover large anonymous
  mappings that bypass the allocator. The event schema (`addr`, `size`,
  `stack`, `timestamp`) generalizes naturally.

---

## 1. Geometric sampling

Per-thread byte counter `next_sample_bytes`. On every allocation of size
`s`:

```rust
fn on_alloc(size: usize) {
    // i64: must be signed — subtracting a large `size` from a small
    // remaining counter must go negative, not wrap around.
    let remaining = next_sample_bytes.get() - size as i64;
    if remaining > 0 {
        next_sample_bytes.set(remaining);
        return;               // fast path: one sub, one branch, done
    }

    // Sampled. Draw a new gap — loop in case the draw is smaller than
    // `-remaining` (rare, but possible for tiny rates / huge allocs).
    let mut next = remaining;
    while next <= 0 {
        next += draw_exponential(sample_rate_bytes);
    }
    next_sample_bytes.set(next);

    record_sample(size, capture_stack());
}
```

Two important details in this structure:

- The `while` loop is important. A single huge allocation (or a very low
  sample rate) can push `next_sample_bytes` below zero by more than one
  draw's worth. We need to keep drawing until the counter is positive
  again so we don't sample the *next* allocation immediately too.
- `remaining` has to be `i64` (or signed `isize`). Subtracting `usize`
  from a `usize` and comparing to zero invites wraparound bugs.

Default `sample_rate_bytes = 512 KiB`. At that rate, a service doing 1 GB/s
of allocation generates ~2000 samples/sec — plenty of signal, trivial
overhead.

### RNG

**Per-thread RNG in TLS, not process-global.** Sampling decisions run
on the allocation hot path — even the *unsampled* path hits this RNG
once per allocation (to decrement the per-thread sample counter and
check whether it tripped). A shared `AtomicU64` would force a
read-modify-write across threads on every allocation; TLS gives each
thread its own state with a plain load/store.

**Reuse the task-dump sampling primitives.** The `taskdump` feature
already implements geometric/Poisson sampling for the same reasons
(see `dial9-tokio-telemetry/src/task_dumped.rs`): a `SplitMix64` PRNG,
a signed `next_sample_ns` counter, a `draw_exponential_ns(mean)`
helper, and a `rng_seed: Option<u64>` config knob with a per-id
timestamp fallback for production uniqueness. The shape is identical
to what we want here — only the unit changes (`bytes` instead of
`nanoseconds`).

The implementation step is to **extract those primitives** from
`task_dumped.rs` into a shared `pub(crate)` module
(e.g. `dial9-tokio-telemetry/src/sampling.rs`) and have both the
existing task-dump code and the new memory profiler import from it.
Concretely the shared API:

```rust
// dial9-tokio-telemetry/src/sampling.rs
pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self;
    pub(crate) fn next_u64(&mut self) -> u64;
    /// Draw from exponential distribution with the given mean.
    /// Always returns at least 1 to avoid immediate re-trigger.
    pub(crate) fn draw_exponential(&mut self, mean: u64) -> i64;
}
```

Memory profiling stores per-thread state in TLS:

```rust
thread_local! {
    static SAMPLE_STATE: Cell<SamplingState> = Cell::new(
        SamplingState::new(global_seed().wrapping_add(thread_nonce()))
    );
}

struct SamplingState {
    next_sample_bytes: i64,
    rng: SplitMix64,
}
```

Each thread seeds its state lazily on first sample from a shared
install-time seed mixed with a per-thread nonce (e.g.
`ThreadId::as_u64()` or a counter), so the stream is deterministic
given `rng_seed` and reproducible across runs for tests.

Short-lived threads that never sample pay zero: `thread_local!`
initializers don't run until the first access, so a blocking-pool
thread that never allocates enough to trip the sampler never
materializes a generator.

`draw_exponential` already implements `-ln(uniform) * mean` via a
clamped `f64` path. That's slightly slower than the `fastlog2` trick
Go uses (`fastexprand` in `runtime/malloc.go`), but it's only on the
*sampled* path (~1 in 8000 allocs at default rate), where stack
capture (~110 ns) dominates the ~10 ns FP cost. If profiling later
shows the sampled path bottlenecking on `f64::ln`, we can swap in
the fast-log2 trick; until then, code reuse wins.

Note: the throwaway `mem-profile-experiments/` crate uses
`Xoshiro256PlusPlus` because it has its own copy of the sampling
math and predates this consolidation. The production implementation
should use the existing `SplitMix64` from the task-dump code path.

### Why geometric over alternatives

Unbiased estimates of total bytes allocated per call site. Reservoir
sampling biases against allocations that occur late in a program's
lifetime. Fixed N-of-every-M biases against large objects that get
undersampled. Jemalloc's profiling internals doc (referenced in the
doc_internal tree) has the full variance analysis — short version: per-byte
Bernoulli sampling via the geometric/exponential trick gives the lowest
variance estimator of the simple strategies.

### Small-object unbiasing

When you sample at rate `R` and see an allocation of size `s < R`, the
*expected* size represented is `R / (1 - exp(-s/R))`. The analysis toolkit
applies this to produce unbiased byte totals. Jemalloc uses the same
formula (`jeprof` divides by `1 - exp(-Z/R)`). Aggregation must happen
*after* unbiasing per sample — sum-then-unbias underreports small-object
stacks. We'll document this for the toolkit.

---

## 2. Stack capture via `perf-self-profile`

The CPU profiler's frame-pointer unwinder (`fp_profiler::unwind::unwind`)
is exactly what we need:
- No allocations.
- Safe against corrupted frame chains (the `safe_load` SIGSEGV handler
  is already installed by `install_handler()`).
- **~5 ns per frame, ~110 ns for a 20-frame walk** on x86_64 (measured
  on AMD EPYC 9R14 via `perf-self-profile/benches/unwind.rs` with
  `-C force-frame-pointers=yes`; 27-frame walk = 146 ns mean, 12-frame
  walk = 66 ns mean). Add ~50–200 ns in production for cold caches;
  faulting frames that hit the SIGSEGV safe-load path cost ~1–5 µs
  for the single faulting frame.

`fp_profiler` is currently `pub(crate)`. We need to expose a public API.

### Proposed public `unwinder` module in `dial9-perf-self-profile`

```rust
// dial9-perf-self-profile/src/lib.rs
pub mod unwinder;
```

```rust
// dial9-perf-self-profile/src/unwinder.rs

/// Handle that proves the SIGSEGV fault handler is installed.
/// Zero-sized, freely copyable.
#[derive(Clone, Copy, Debug)]
pub struct Unwinder { _private: () }

impl Unwinder {
    /// Install the SIGSEGV fault handler used by stack capture.
    /// Idempotent: safe to call multiple times from multiple threads.
    ///
    /// Returns `Err` if `sigaction` fails.
    ///
    /// # Requirements
    /// - Frame pointers (build with `-C force-frame-pointers=yes`).
    pub fn install() -> std::io::Result<Self>;

    /// Capture a stack trace of the calling thread into `out`. Returns
    /// the number of frames written. Never allocates.
    ///
    /// # Frame-0 contract
    /// `out[0]` is the return address *into the caller of `capture`* —
    /// i.e. the PC where `capture` itself will return. Subsequent frames
    /// walk outward via the frame-pointer chain. Callers should expect
    /// to skip `capture` itself plus any `#[inline(never)]` shim they
    /// insert.
    ///
    /// In particular: if `capture` is called from a helper
    /// `on_alloc_sampled()` that is in turn called from `GlobalAlloc::alloc`,
    /// frame 0 will be inside `on_alloc_sampled`, frame 1 inside
    /// `GlobalAlloc::alloc`, frame 2 at the user allocation site. The
    /// analysis toolkit symbolizes all frames and the viewer presents
    /// them unmodified; UI-level "skip frames for clarity" stays a UI
    /// decision, not a capture-time one.
    ///
    /// # Safety contract
    /// Must not be called from inside a different SIGSEGV handler.
    pub fn capture(&self, out: &mut [u64]) -> usize;
}
```

**Known bias: `MAX_FRAME_SIZE`.** The underlying unwinder stops walking
if `saved_fp - fp > 256 KiB` (this cap rejects wild pointers that happen
to be above `fp` but aren't real frames). For CPU profiling the
occasional truncation caused by a large on-stack future or struct is
rare enough to ignore; for *allocation* profiling the bias is
systematic — allocations performed inside functions with unusually
large stack frames (large `Box::pin(future)` state machines, large
`[u8; N]` locals) will consistently have their stacks cut off at the
big frame.

**Why not just raise the cap now?** The 256 KiB threshold is a
safety/correctness tradeoff, not a performance one. A higher cap means
the unwinder follows more wild pointers before giving up, increasing
the chance of reading garbage memory (triggering SIGSEGV safe-load
faults, ~1-5µs each) or producing bogus frames. We'd need to validate
a higher cap empirically across real workloads to confirm it doesn't
degrade stack quality. Plan: raise to 1 MiB in the unwinder PR (step 1
of rollout) with a benchmark that measures false-frame rate at the
higher cap. If the false-frame rate stays negligible, ship it; if not,
keep 256 KiB and document the limitation.

**Why a handle-returning `install`, not auto-install-on-first-capture?**

1. Install happens once, at a point the caller chooses. No hidden
   first-call latency spike buried inside `capture`.
2. Install can fail (`sigaction` → `EINVAL` on weird kernels, or a
   constructor conflict). A `Result` at install time surfaces this
   cleanly; a fallible `capture` would have to decide between silently
   returning 0 or propagating an error on the hot path — neither is
   good.
3. `&self` gives `capture` a natural place to hang per-unwinder state
   later (thread-local caches, config) without a breaking API change.
4. The "handler installed" check disappears from the hot path. Holding
   an `Unwinder` is the proof.

Internally `capture` does:
1. Read `pc`, `fp`, `sp` of the current frame via inline asm.
2. Call the existing `unwind(pc, fp, sp, out)`.

The current `unwind_from_ucontext` path stays internal — it's only
needed from inside a signal handler, which external callers shouldn't
be doing.

`MemoryProfiler::build` calls `Unwinder::install()` and stores the
returned handle. The hook accesses it via the process-global
`MemoryProfilerInner` (§8):

```rust
struct MemoryProfilerInner {
    unwinder: Unwinder,
    ...
}

// In the hook:
// 128 frames × 8 B = 1 KiB stack buffer. Rust async call stacks routinely
// exceed 40 frames (state machines, tower layers, hyper service stacks,
// futures::join_all), so 64 is on the edge; 128 gives comfortable
// headroom without meaningful stack pressure.
let mut frames = [0u64; 128];
let n = inner.unwinder.capture(&mut frames);
```

### SIGSEGV handler lifecycle

The handler is permanently installed after `Unwinder::install()`.
It is **not** uninstalled when capture completes. That's safe and
intentional:

- The divert-and-return logic fires *only* when the faulting PC falls
  within the `safe_load_start..safe_load_end` code range. Those
  instructions execute only inside the `unwind` walk. Everywhere else
  in the process, a SIGSEGV has a PC outside that range and the
  handler chains to the previously-installed one (or to `SIG_DFL`,
  which terminates the process as expected).
- "Uninstall between captures" would just be
  `sigaction(SIGSEGV, old, null)` and `sigaction(SIGSEGV, ours, old)`
  around every `capture`. That's two syscalls per stack walk
  (hundreds of ns each) — an order of magnitude more expensive than
  the ~110 ns walk itself — for no correctness gain.

**Known limitation:** if the application installs its own SIGSEGV
handler *after* dial9 initializes, it will not chain back to ours,
and `safe_load` loses its fault tolerance. This is a pre-existing
issue with CPU profiling and applies identically here. Applications
that install their own SIGSEGV handlers should do so before
initializing dial9.

---

## 3. Events

### `AllocEvent`

```rust
#[derive(TraceEvent)]
struct AllocEvent {
    #[traceevent(timestamp)]
    timestamp_ns: u64,
    /// OS thread ID. Matches CpuSample.tid.
    tid: u32,
    /// Allocation size in bytes.
    size: u64,
    /// Returned pointer. Only meaningful when liveset tracking is on —
    /// otherwise 0. Field always present so the schema is stable.
    addr: u64,
    /// Stack at the allocation site.
    stack: InternedStackFrames,
}
```

**Where does the sampling rate live?** Not on the event. The sample
rate is **immutable for the life of a trace file** and is written
into `SegmentMetadata` (the same mechanism used by `RuntimeName`,
`BootId`, etc.):

```
SegmentMetadata["memory_profile.sample_rate_bytes"] = "524288"
```

The analysis toolkit reads `sample_rate_bytes` from segment metadata
and applies it uniformly to every `AllocEvent` in that segment when
computing unbiased byte totals. Per-event storage of the weight
would add 1–9 LEB128 bytes to every sampled allocation for data
that never varies within a file — wasteful at steady-state trace
rates.

**Can the rate change at runtime?** Not within a single trace file.
If we later want a `set_sample_rate_bytes` API, it will:

1. Atomically update the rate held in `MemoryProfilerInner`.
2. Force a segment rotation so the new segment's `SegmentMetadata`
   carries the new rate.
3. Existing per-thread `next_sample_bytes` counters drain naturally
   under the old rate; subsequent redraws use the new rate. The
   tiny post-change bias (at most one gap's worth per thread using
   the old rate) is negligible.

For the MVP, the rate is set once at `install()` and never changes.

**Why explicit `tid`?** A trace can contain allocations from threads
that aren't currently inside a poll — blocking-pool workers,
user-spawned OS threads, early-boot allocations before any runtime
exists. `CpuSample` already carries `tid` for the same reason; we
match, so alloc events and CPU samples join cleanly on thread.

We **don't** carry a `worker_id`: when the alloc happens on a tokio
worker, the worker's identity is recoverable by joining `tid` to the
most recent `WorkerUnparkEvent.tid` ≤ `AllocEvent.timestamp_ns` for
that worker. Keeping `worker_id` off the event shaves bytes per sample
in the common case, avoids inventing a sentinel for "not on a worker,"
and — more importantly — keeps the allocator hook decoupled from
tokio runtime state. The hook captures only `tid` (cheap; uses the
existing `current_tid()` helper that returns a real `gettid()` on
Linux/Android and a synthetic per-process counter elsewhere); the
viewer/toolkit do the join.

> **Pre-requisite for the join to work**: `WorkerParkEvent` /
> `WorkerUnparkEvent` must include `tid`. That's a small additive
> change to those event schemas listed as a pre-step in the §15
> rollout. Today they only carry `worker_id` (no `tid`), which is
> why the existing CPU profiler resolves `tid → worker_id` at flush
> time via `SharedState::thread_roles`. Once park/unpark carry
> `tid`, this resolution can move to analysis time and the trace
> becomes self-describing for thread-role attribution. (As a bonus,
> we also gain visibility into the `block_in_place` case, where a
> single `worker_id` is backed by different `tid`s over its
> lifetime.)

`ThreadNameDef { tid, name }` already exists in the format (emitted
by the CPU profiler) and does double-duty for alloc events — no new
per-thread metadata needed.

### `FreeEvent` (liveset only)

```rust
#[derive(TraceEvent)]
struct FreeEvent {
    #[traceevent(timestamp)]
    timestamp_ns: u64,
    /// OS thread ID of the thread that ran the dealloc. Carried for
    /// trace reconstruction: lets analysis attribute the free to a
    /// poll/task by joining against `WorkerPark`/`WorkerUnpark` and
    /// `PollStart`/`PollEnd` events, the same way `AllocEvent.tid`
    /// works.
    tid: u32,
    /// Pointer that was freed. Matches a previously-seen `AllocEvent.addr`.
    addr: u64,
    /// Size of the allocation being freed. Denormalized from the
    /// matching `AllocEvent` so the free stays analytically useful
    /// when the corresponding `AllocEvent` has been evicted by trace
    /// rotation.
    size: u64,
    /// Monotonic-ns timestamp of the original `AllocEvent`. Allows
    /// leak analysis to bucket frees by generation without needing
    /// the `AllocEvent` in the same (unrotated) trace.
    alloc_timestamp_ns: u64,
}
```

**Why denormalize `size` and `alloc_timestamp_ns`?** `RotatingWriter`'s
default is 60-second segments with a total-size eviction budget. A
long-lived allocation (hours of cache residency) will have its
`AllocEvent` in a segment that has been evicted by the time its
`FreeEvent` is written. A free that carries only `addr` is then
unjoinable — no size to subtract from live-heap totals, no
timestamp to bucket by generation. Denormalizing keeps `FreeEvent`
self-sufficient for net-bytes-freed and generational leak analysis
across rotation.

The hot-path cost is zero: the liveset entry we look up to decide
whether to emit the free already carries both fields. Wire cost is
a few bytes per `FreeEvent` (LEB128 varint encoding).

We do **not** denormalize the allocation stack onto `FreeEvent`:
storing the full stack in every liveset entry would bloat the liveset
~8× (64 × 8 B per stack vs the ~16 B size+timestamp pair), and that
memory is paid *while the allocation is live*. Stack-attributed leak
analysis is still possible inside a single retained segment via the
`AllocEvent↔FreeEvent` join on `addr`.

`FreeEvent` carries `tid` (matching `AllocEvent`) so analysis can
attribute the free to a poll/task by joining against the runtime's
`WorkerUnpark`/`PollStart` history — the same join used for
`AllocEvent` and `CpuSample`. We don't carry an explicit `worker_id`:
when the free happens on a tokio worker, the worker's identity is
recoverable from `tid` plus the recent `WorkerUnpark` for that
thread, the same way the alloc side handles it (§3 — *Why explicit
`tid`?*).

### `realloc` handling

What other tools do:

**Go runtime**: treats realloc as a normal `malloc` of the new size
plus a `free` of the old pointer. Both sides are subject to independent
sampling. (`mallocgc` handles reallocation in the user-visible
`append`/`growslice` paths; the profile sees it as two events.)

**jemalloc `prof_realloc`**: 4-way combination based on which sides
were sampled:
- Old sampled + new sampled: emit free-of-old, emit alloc-of-new.
- Old unsampled + new sampled: emit alloc-of-new only.
- Old sampled + new unsampled: emit free-of-old only.
- Neither sampled: emit nothing.

Crucially, jemalloc treats the new side as a *fresh* sampling decision
even when `realloc` doesn't move the pointer. Same as a new `malloc`.

**Our plan: follow jemalloc.** Treat `realloc(p, n_bytes)` as:
1. `dealloc(p, old_layout)` — may emit `FreeEvent` if `p` was sampled
   (and liveset tracking is on).
2. `alloc(new_layout)` — fresh sampling decision, may emit `AllocEvent`.

This is symmetric, simple, and matches existing tooling conventions.
In-place realloc (pointer unchanged) still goes through the same flow;
we don't try to be clever about it. That means in rare cases we might
emit `FreeEvent { addr: p }` followed by `AllocEvent { addr: p }` for
the same pointer — fine, the timestamps disambiguate.

The `Dial9Allocator::realloc` impl delegates to
`self.0.realloc(ptr, old_layout, new_size)` and runs the
alloc/dealloc hook logic around it.

### Worker / task attribution for polls

Both events carry an implicit task_id via the shared `PollStart` context
the flush pipeline already builds. We do **not** add an explicit
`task_id` field because:
1. Every alloc inside a poll already falls between that worker's most
   recent `PollStart` and the matching `PollEnd`.
2. The analysis toolkit already uses this range-matching for CPU samples
   (see `dial9-viewer/skills/analyze.js`).

Allocations outside any poll — from non-worker threads, or on worker
threads that aren't currently polling — carry only `tid` and get no
task attribution. The viewer joins `tid` against the runtime's
`WorkerUnpark` history to decide whether the allocation was on a
tokio worker at that instant, and shows out-of-poll allocations in a
"blocking" or "unknown" lane when not.

---

## 4. `Dial9Allocator<A>`

Generic wrapper, default `A = System`:

```rust
pub struct Dial9Allocator<A = std::alloc::System>(A);

impl Dial9Allocator {
    /// Wrap the system allocator. Use this when you don't need a
    /// custom inner allocator (i.e. you weren't otherwise setting
    /// `#[global_allocator]`).
    pub const fn system() -> Self { Self(std::alloc::System) }
}

impl<A: GlobalAlloc> Dial9Allocator<A> {
    /// Wrap a custom allocator (e.g. jemalloc, mimalloc).
    pub const fn new(inner: A) -> Self { Self(inner) }
}

unsafe impl<A: GlobalAlloc> GlobalAlloc for Dial9Allocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { self.0.alloc(layout) };
        if !ptr.is_null() {
            hook::on_alloc(ptr, layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        hook::on_dealloc(ptr, layout.size());
        unsafe { self.0.dealloc(ptr, layout) };
    }

    unsafe fn realloc(
        &self, ptr: *mut u8, old_layout: Layout, new_size: usize
    ) -> *mut u8 {
        let new_ptr = unsafe { self.0.realloc(ptr, old_layout, new_size) };
        if !new_ptr.is_null() {
            // Only record the free-of-old after confirming realloc
            // succeeded. If realloc returns null, the old pointer is
            // still live and must not be recorded as freed.
            hook::on_dealloc(ptr, old_layout.size());
            hook::on_alloc(new_ptr, new_size);
        }
        new_ptr
    }

    // alloc_zeroed delegates similarly.
}
```

User code:

```rust
// Common case: system allocator. No turbofish needed.
#[global_allocator]
static ALLOC: Dial9Allocator = Dial9Allocator::system();

// Wrapping a custom allocator:
#[global_allocator]
static ALLOC: Dial9Allocator<tikv_jemallocator::Jemalloc> =
    Dial9Allocator::new(tikv_jemallocator::Jemalloc);
```

The wrapper is zero-cost when memory profiling isn't attached (see §8).

---

## 5. Ring-buffer hand-off (the chosen design)

The allocator hook does the **bare minimum** on the allocating thread —
sampling decision, stack capture, push a fixed-size POD record into a
process-global lock-free queue. A dedicated **consolidator thread** drains
the queue and turns each record into the corresponding `AllocEvent` /
`FreeEvent` in the trace.

This is *the* design tension this whole doc dances around: everything an
allocator hook touches must itself be allocator-quiet. Encoders allocate.
Concurrent hashmaps allocate. `tracing::warn!` allocates. `Vec::with_capacity`
allocates. Each of those becomes a potential deadlock or recursion when run
from inside `GlobalAlloc::alloc`. A ring-buffer hand-off makes that whole
class of problem go away by construction:

- The hook never holds a lock that any other code path could allocate
  inside of.
- The hook never calls into a data structure that has its own thread-local
  state (papaya's `seize`, encoder's location cache, etc.) — anything with
  TLS becomes a thread-shutdown footgun.
- The consolidator thread *opts out* of being sampled (one TLS flag, set
  at thread spawn). It is then free to allocate, lock, intern, and grow
  hashmaps without worrying about pushing more samples back into the ring.

### What the allocator hook does

1. Read process-global `OnceLock<RingProfilerState>`. Branch on null.
2. Subtract `size` from the per-thread `next_sample_bytes` counter and
   compare to zero (~1 ns).
3. **Unsampled (~99.9% of allocs):** return.
4. **Sampled:**
   - `Unwinder::capture(&mut frames)` into an on-stack `[u64; MAX_FRAMES]`
     (~110 ns for ~20 frames; see §2).
   - Build a `RawSample { tid, size, ts_ns, addr, frames, frame_count }`
     on the stack (~1 KiB fixed-size record).
   - `queue.push(sample)` — lock-free MPMC bounded-queue push (~10–30 ns
     uncontended on x86_64).
   - On queue full: increment a dropped-samples counter, continue.

That's it. No mutex. No encoder. No liveset hashmap. No `tracing::warn!`.
No `Vec::with_capacity`.

### What the consolidator thread does

The "consolidator" isn't a new thread — it's the **existing dial9 flush
thread** that already drains `CpuProfiler` and `SchedProfiler` every
5 ms. We register the ring as one more `Source` (§6) and the flush
thread picks it up. The flush loop becomes (sketch):

```rust
loop {
    for source in &recorder.sources {
        source.flush(&ctx);              // see §6 ("Source trait")
    }
    buffer::drain_to_collector(...);     // unchanged
    park_timeout(Duration::from_millis(5));
}
```

`Source::flush` pops every pending `RawSample` and:

1. Interns the stack via the consolidator thread's own
   `ThreadLocalEncoder` (this is just `record_encodable_event` from the
   user-events path — same code that all other dial9 events go through).
2. Calls `encoder.encode(&AllocEvent { … })`. The encoded batch lands
   in the consolidator thread's TL buffer and flows into
   `CentralCollector` exactly like any other event.
3. For `RawSample::Free` records, looks up `addr` in a `HashMap<usize, LivesetEntry>`
   that lives **only** in the consolidator (single-threaded — no `Mutex`,
   no `papaya`, no `crossbeam-skiplist`). Hit → emit `FreeEvent`. Miss →
   ignore.

The consolidator's allocations *do* re-trigger the hook, but that's
fine: the hook on the flush thread runs the same sampling decision
as any other thread, samples a small fraction of its own allocations
(~0.01% of trace events at default rate), and pushes them into the
ring like any other sample. The push is allocator-quiet so it can't
cascade. See §7 for why this is intentional and how the geometric
self-sampling stays bounded.

### Why MPMC `crossbeam_queue::ArrayQueue`?

- Producers come from arbitrary threads (tokio workers, blocking pool,
  user OS threads, library-spawned threads, the std library's
  `set_hook`-spawned panic thread, etc.). A per-thread SPSC ring would
  require per-thread setup that itself allocates — defeating the point.
- `ArrayQueue` is wait-free for `push` (CAS on a single tail slot index)
  and the workspace already depends on `crossbeam-queue` (used by
  `BoundedQueue` in `CentralCollector`).
- Contention concern: at the **default** 512 KiB sample rate and 1 GB/s
  allocation rate, the system pushes ~2K samples/sec. Even on a 32-core
  machine that's ~60 contended pushes per second per core — invisible.
  Stress-test results below confirm the queue holds up under
  pathologically high sample rates too.

### Sizing the ring

`MAX_FRAMES = 64` (1 KiB stack budget for the on-stack capture buffer).
`DEFAULT_RING_CAPACITY = 4096` slots → 4096 × 520 B ≈ **2.1 MiB**
process-global allocation, paid up front at install time. That's enough
for **2.0 seconds** of accumulated samples at the 2K/sec target rate, or
~40 ms at a stress-test 100K/sec rate. The consolidator drains every
5 ms, so the ring should never see more than a few dozen entries in
realistic operation.

If the queue does fill (consolidator stuck, producers spike), pushes
fail and the producer increments the per-source `dropped_samples`
counter. Not catastrophic — the next flush emits a `dial9_stats`
event with the count, and we keep moving.

### Measurements

These come from `mem-profile-experiments/` (excluded from the workspace,
contains both implementations side by side). Release build, x86_64,
AMD EPYC class CPU, frame-pointers enabled, `MAX_FRAMES = 128`.

**Hot-path cost per allocator call:**

| Bench | Description | Time |
|---|---|---|
| `baseline_sampling_only` | Just the sampling counter — lower bound | 2.60 ns |
| `precise_unsampled` | Inline design, unsampled fast path | 3.57 ns |
| `ring_unsampled` | Ring design, unsampled fast path | 1.10 ns |
| `precise_mixed_1in1024` | Inline, 1-in-1024 sampled mix | 26.34 ns |
| `ring_mixed_1in1024` | Ring, 1-in-1024 sampled mix | 33.82 ns |
| `precise_sampled_deep20` | Inline, force-sampled, depth-20 stack | 1058 ns |
| `ring_sampled_deep20` | Ring, force-sampled, depth-20 stack | 1226 ns |

**Dealloc-path cost per allocator dealloc call (liveset on):**

| Bench | Description | Time |
|---|---|---|
| `dealloc_baseline` | No-op anchor (no profiling at all) | 0.27 ns |
| `ring_dealloc_uncontended` | Single producer + drainer, push `RawFree` | 9.27 ns |
| `ring_dealloc_contended (8 threads)` | 8 producers fighting for the queue | ~3 µs/push |

**Consolidator-side cost (flush thread):**

| Bench | Description | Time |
|---|---|---|
| `consolidator_rate_limited` | 8 producers @ ~250 samples/s/thread | 6.79 ns |
| `consolidator_busy_producers` | 8 producers flat-out (stress test) | 308 ns |
| `consolidator_drain` (4 producers @ 1 KiB rate) | Saturated, drain per pop | ~725 ns |

**Takeaways (revised after re-running with `MAX_FRAMES = 128`):**

1. **Unsampled fast path:** ring is still ~3× faster (1.1 ns vs 3.57 ns).
   This is what dominates real workloads — at 1 GB/s allocation, ~99.9%
   of allocs are unsampled, so the unsampled-path cost wins by 2.5 ns × 1M
   allocs/sec/thread = ~2.5 ms/sec/thread of CPU recovered.
2. **Sampled path:** the 128-frame `RawAlloc` record (~1 KiB) is *slower*
   to push than the precise design's inline encode (1226 ns vs 1058 ns).
   This is the cache-line cost the ring pays for richer stacks. The
   amortized impact is small: at 2K samples/sec, that's ~340 µs/sec of
   extra CPU on producer threads — invisible in practice.
3. **Mixed-path bench is misleading at 1-in-1024 sample rate.** With one
   sampled alloc per 1024 unsampled, the sampled-path cost dominates the
   average (1226 / 1024 ≈ 1.2 ns per call from the sampled side alone).
   At realistic 1-in-8192 (default 512 KiB rate, 64-byte allocs), the
   ring's 1.1-ns unsampled path dominates and the average drops back to
   ~1.3 ns.
4. **Dealloc path is the new concern.** Pushing `RawFree` records on every
   dealloc costs 9.3 ns uncontended but the MPMC queue saturates near
   2.7 M pushes/sec aggregate under heavy contention. For services with
   sustained 10M+ deallocs/sec we'll need a per-thread free-buffer
   batch (see §9). Default is liveset-off, so most users never pay this.
5. **Consolidator throughput:** at realistic 2K/sec sample load,
   `Source::flush` costs ~7 ns per pop. A flush cycle every 5 ms drains
   maybe a dozen entries — invisible.
6. **Beyond performance:** the ring design is *radically simpler* to
   reason about. The WIP branch's hot path needed `try_with`, `try_lock`,
   `catch_unwind`, a spare-buffer dance, and a `ReentrancyGuard`
   straddling the encoder's mutex scope. The ring design needs
   `try_with` on the sampling state TLS and `OnceLock::get()` — that's
   it.

(Reproducing the numbers: `cd mem-profile-experiments && cargo bench`.
The crate is intentionally outside the workspace and gets thrown away
once the design lands; until then it documents the comparison.)

---

## 6. The `Source` trait

A second motivation for moving memory profiling to a ring/consolidator
shape: the existing `CpuProfiler` and `SchedProfiler` already follow
that pattern. The flush thread today does (in `EventWriter::flush_cpu`):

```rust
profiler.drain(|raw, thread_name| {
    record_encodable_event(&CpuSampleData { … }, &shared.collector, &shared.drain_epoch);
});
```

The memory profiler's consolidator does the same shape: pop from a queue,
call `record_encodable_event` for each. The only difference is the
producer is the allocator hook instead of perf events.

We extract this pattern into a `Source` trait:

```rust
pub(crate) trait Source: Send + Sync {
    /// Drain pending data into the dial9 trace. Called once per flush
    /// cycle from the flush thread. Implementations are responsible for
    /// synchronization with their own producers (allocator hooks, perf
    /// samplers, etc.).
    fn flush(&mut self, ctx: &FlushContext<'_>);

    /// Best-effort count of pending items. May be approximate. Used for
    /// metrics and backpressure decisions.
    fn pending(&self) -> usize { 0 }

    /// Diagnostic name. Used for log messages and metrics keys
    /// (e.g. `"memory"`, `"cpu_profile"`, `"sched"`).
    fn name(&self) -> &'static str;
}
```

### `FlushContext` definition

The context bundles exactly the bag of references that
`EventWriter::flush_cpu` already passes around today as a tuple of
arguments:

```rust
pub(crate) struct FlushContext<'a> {
    /// Where flushed batches go. `record_encodable_event` writes here
    /// via the flush thread's own `ThreadLocalBuffer`.
    pub(crate) collector: &'a Arc<CentralCollector>,
    /// Drain epoch. Bumped by the flush thread to wake silent
    /// thread-local buffers on the producer side. Sources that emit
    /// many events in a single `flush` call should pass this through
    /// to `record_encodable_event` so their work is batched correctly.
    pub(crate) drain_epoch: &'a AtomicU64,
    /// `tid → ThreadRole` mapping snapshot. CPU profiler and memory
    /// profiler both use this to resolve worker IDs from raw OS thread
    /// IDs at flush time. Snapshotted once per flush cycle to avoid
    /// per-event lock contention on `SharedState::thread_roles`.
    pub(crate) thread_roles: &'a HashMap<u32, ThreadRole>,
}
```

`FlushContext` is constructed once per cycle by the flush thread and
passed to every registered `Source`. Lifetimes: the context borrows
from the flush thread's stack and from `SharedState`, both of which
outlive any single `flush` call. The `'a` parameter just keeps the
compiler honest about that scope.

### `Source` impl sketches

**`CpuProfiler`** today (cpu_profile.rs:120) takes a closure with the
signature `FnMut(RawCpuSample, Option<&ThreadName>)`. Migrating to
`Source` moves the closure body — currently in
`EventWriter::flush_cpu` — into the impl:

```rust
impl Source for CpuProfiler {
    fn flush(&mut self, ctx: &FlushContext<'_>) {
        let resolve = |tid: u32| -> WorkerId {
            match ctx.thread_roles.get(&tid) {
                Some(ThreadRole::Worker(id)) => WorkerId::from(*id),
                Some(ThreadRole::Blocking) => WorkerId::BLOCKING,
                None => WorkerId::UNKNOWN,
            }
        };
        self.sampler.for_each_sample(|sample| {
            // Filter child-process samples (perf inherit leak).
            if sample.pid != self.pid { return; }
            // Cache thread name for non-worker tids.
            self.tid_to_name
                .entry(sample.tid)
                .or_insert_with(|| read_thread_name(sample.tid).map(ThreadName::new).unwrap_or_default());
            let data = CpuSampleData {
                timestamp_nanos: sample.time,
                worker_id: resolve(sample.tid),
                tid: sample.tid,
                source: CpuSampleSource::CpuProfile,
                callchain: sample.callchain.clone(),
                thread_name: self.tid_to_name.get(&sample.tid).cloned(),
                cpu: sample.cpu,
            };
            record_encodable_event(&data, ctx.collector, ctx.drain_epoch);
        });
    }
    fn pending(&self) -> usize { /* perf ring depth, if cheap */ 0 }
    fn name(&self) -> &'static str { "cpu_profile" }
}
```

The diff from today's code is mechanical: the closure body becomes the
inner expression and the closure parameters (`raw`, `thread_name`)
become local variables. `&mut self` is required because the thread-name
cache is mutated; the recorder wires sources behind a per-source
`Mutex<Box<dyn Source>>` (the same shape `SharedState::sched_profiler`
uses today via `Mutex<Option<SchedProfiler>>`).

**`MemoryProfileSource`** is the *third* implementation:

```rust
pub(crate) struct MemoryProfileSource {
    ring: Arc<ArrayQueue<RawSample>>,
    liveset: Option<HashMap<usize, LivesetEntry>>,
    rate_bytes: u64,
}

impl Source for MemoryProfileSource {
    fn flush(&mut self, ctx: &FlushContext<'_>) {
        // Pop everything pending. Bounded by ring capacity (4096),
        // so this loop is at most ~4096 iterations per flush cycle.
        while let Some(sample) = self.ring.pop() {
            match sample {
                RawSample::Alloc(a) => {
                    let event = AllocEvent {
                        timestamp_ns: a.ts_ns,
                        tid: a.tid,
                        size: a.size,
                        addr: a.addr,
                        callchain: /* intern via ctx, see below */,
                    };
                    record_encodable_event(&event, ctx.collector, ctx.drain_epoch);
                    if let Some(liveset) = self.liveset.as_mut() {
                        liveset.insert(a.addr as usize, LivesetEntry {
                            size: a.size,
                            timestamp_ns: a.ts_ns,
                        });
                    }
                }
                RawSample::Free(f) => {
                    if let Some(liveset) = self.liveset.as_mut() {
                        if let Some(entry) = liveset.remove(&(f.addr as usize)) {
                            let event = FreeEvent {
                                timestamp_ns: f.ts_ns,
                                tid: f.tid,
                                addr: f.addr,
                                size: entry.size,
                                alloc_timestamp_ns: entry.timestamp_ns,
                            };
                            record_encodable_event(&event, ctx.collector, ctx.drain_epoch);
                        }
                    }
                }
            }
        }
    }
    fn pending(&self) -> usize { self.ring.len() }
    fn name(&self) -> &'static str { "memory" }
}
```

Stack interning: `record_encodable_event` calls
`buffer::with_encoder(...)` which provides a `ThreadLocalEncoder` to
the encoder closure. `AllocEvent::encode` calls
`enc.intern_stack_frames(stack)` to intern the captured frames. So the
flush thread's own thread-local encoder is what handles interning, and
its allocations are safe even though they re-trigger the hook: the
hook's ring push is allocator-quiet, so the recursion is bounded by
the geometric sample rate (~0.01% self-pollution; see §7).

> **Decoupling note**: `MemoryProfileSource::flush` uses only
> `ctx.collector` and `ctx.drain_epoch` from `FlushContext` — never
> `ctx.thread_roles`. The hook captures `tid`, the consolidator
> emits `tid` on the wire, and any `worker_id` resolution happens at
> analysis time by joining against `WorkerUnparkEvent.tid` (see §3
> *Why explicit `tid`?*). This keeps the memory profiler's runtime
> dependencies to "the dial9 recorder + the global allocator" — no
> tokio-runtime knowledge required, even transitively. By contrast,
> the existing `CpuProfiler::flush` *does* use `ctx.thread_roles`
> for ergonomic `worker_id` resolution at flush time. Both choices
> are valid; sources opt in to the parts of `FlushContext` they
> actually need.

**What this lets us do:**

- The flush thread iterates over a `Vec<Mutex<Box<dyn Source>>>` and
  calls `source.lock().flush(&ctx)` on each. No conditionally-compiled
  per-source branches in the flush loop.
- New sources (e.g. an `mmap` interception, see Non-goals' evolution
  path) drop in by implementing `Source` and getting registered with
  the recorder's flush set.
- Tests can mock a `Source` to drive the flush loop without spinning
  up perf events or an allocator.

The trait is `pub(crate)` for now. We don't expose user-defined sources
publicly — the existing public surface (`record_event`, custom events
via `#[derive(TraceEvent)]`) covers user data ingestion at a higher
level, and adding a public `Source` API would mean stabilizing a much
larger contract (lifetime of pending data across rotation, recovery on
flush failure, etc.) that we don't want to commit to today.



---

## 7. Reentrancy

A global allocator sees *every* allocation in the process, including
allocations performed by dial9 itself. The precise/inline design
(WIP branch) had to address this with a `ReentrancyGuard` because
the hook acquired a mutex on the encoder buffer and called back into
the encoder, which itself allocated — a real deadlock hazard.

The ring-buffer design eliminates the hazard by construction:

- The hook never holds a lock that any other code path could allocate
  inside of (the `ArrayQueue::push` is lock-free CAS).
- The hook never calls into a data structure with its own thread-local
  state (no papaya, no `seize` epoch reclamation, nothing with
  shutdown-order surprises).
- The hook never grows a `Vec` or hashmap.

The flush thread's own allocations — interning stacks via
`ThreadLocalEncoder`, growing the encoder's hashmaps, inserting into
the consolidator-side liveset `HashMap` — *do* trigger the hook.
That's fine and intentional:

- The hook on the flush thread does the same sampling decision as
  any other thread (~1 in 8000 allocs at default rate).
- Sampled allocations from the flush thread push a `RawAlloc` into
  the ring, just like any other thread's samples.
- The push is allocator-quiet, so it doesn't re-trigger the hook.
- The next flush cycle drains those self-samples into `AllocEvent`s
  with stacks rooted in `dial9_tokio_telemetry::*` frames.

This is **a feature, not a bug**: dial9's own allocation pressure
shows up in the trace alongside the user's. If the profiler itself
is generating noticeable allocator traffic (encoder grows, hashmap
rehashes, batch buffer reallocs), that's information we want — it
tells us where to optimize the implementation. Filter the
`dial9_tokio_telemetry::*` frames out at analysis time if a "pure
user-code" view is needed.

### Geometric self-sampling is bounded

It's worth working through why the self-sampling doesn't run away:

1. The flush thread drains N producer samples per cycle.
2. Encoding/interning those samples does ~kN allocations on the flush
   thread (k is small — a few allocations per emitted event).
3. ~kN/8000 of those trip the sampling counter and produce one ring
   push each.
4. The next flush cycle drains those kN/8000 self-samples, plus a new
   batch of producer samples; the self-samples generate
   ~k(kN/8000)/8000 = k²N/8000² second-order self-samples.
5. Geometric series converges fast. Steady-state self-sampling rate
   is ~k/8000 (≈0.01% of trace samples at default sample rate).

This is small enough that it's not a concern for trace size, viewer
performance, or analysis correctness. Users who need a "pure
user-code" view filter the `dial9_tokio_telemetry::*` frames at
analysis time.

### Specifically, the WIP's papaya/seize TLS panics are eliminated

The WIP branch wrapped every `liveset.insert` / `liveset.remove` call
in `catch_unwind` (see `memory-profiling-wip` `hook.rs:70`) because
papaya's internal `seize` epoch reclamation has its own per-thread
state. During thread shutdown, that state can be destroyed before our
own TLS, panicking the `insert`/`remove` call from inside the hook.

The ring design eliminates this by construction: the consolidator-side
liveset is a plain `std::collections::HashMap` (§9), accessed by
exactly one thread (the flush thread). No epoch reclamation, no
shared TLS, nothing to panic. The producer side never touches the
liveset at all — it just pushes `RawFree` records into a queue.

### Allocation during stack capture

`Unwinder::capture()` must never allocate. We enforce this by only
using stack-resident buffers (`[u64; MAX_FRAMES]`) and calling into
the existing fp-based unwinder, which is allocation-free. This was
true for the WIP design too; carrying it forward.

### Early-boot allocations

Allocations happen before `main` runs (`static` initializers, argv
parsing). At that point no `MemoryProfiler` exists. The hook checks
the `OnceLock<MemoryProfilerState>` (§8) and returns early if unset.
~1 ns overhead per early-boot alloc. No special handling needed.

---

## 8. Configuration — static install, captured handle

`MemoryProfiler::install()` sets a process-global static that the
`Dial9Allocator` reads on every allocation. Installation can happen
**exactly once per process** and takes a `TelemetryHandle` captured
at install time. The hook uses that captured handle directly —
no `TelemetryHandle::current()` lookup on the hot path.

This works for any thread, not just tokio workers. When a non-tokio
thread performs its first sampled allocation, the hook pushes a
`RawSample` into the global ring. The flush thread later drains the
ring and emits the corresponding `AllocEvent` via its own
`ThreadLocalEncoder` and `record_encodable_event` — same code path
that `tracing_layer` or custom events go through. A random
`std::thread::spawn`ed thread that allocates → ring → flush thread →
central collector → trace file.

### Shape

```rust
use dial9_tokio_telemetry::memory_profiling::{
    Dial9Allocator, MemoryProfiler, TimestampMode,
};

// Install the global allocator (static, runs before main).
#[global_allocator]
static ALLOC: Dial9Allocator = Dial9Allocator::system();

fn main() {
    // 1. Build the runtime as usual.
    let guard = TelemetryCore::builder()
        .writer(writer)
        .trace_path("/tmp/trace.bin")
        .build()?;
    guard.enable();

    // 2. Install the memory profiler with the live handle.
    //    Returns Err(AlreadyInstalled) if called a second time.
    let _mem = MemoryProfiler::builder()
        .sample_rate_bytes(512 * 1024)
        .track_liveset(true)
        .timestamp_mode(TimestampMode::ReusePollStart)
        .install(guard.handle())?;

    // 3. Build + attach a runtime. Allocations on every thread —
    //    including non-tokio threads — are captured.
    let (rt, _) = guard.trace_runtime("main").build(rt_builder)?;
    rt.block_on(async { /* ... */ });
}
```

Between steps 1 and 2 (after the global allocator static is installed
but before the profiler is configured), every allocation takes the
unset-`OnceLock` fast path: one `Acquire` load + null check (~1ns),
not set → skip. No events, no crashes, no setup order to get right.

### Why capture the handle, not look it up per-alloc

The `Dial9TokioLayer` calls `TelemetryHandle::current()` on each
event because tracing spans run on arbitrary threads and the layer
needs to discover whether the *current* thread belongs to a dial9
runtime. The layer has no concept of "the" runtime — if multiple
runtimes run concurrently, each thread's events flow into its own
runtime's trace.

For memory profiling, that's the wrong semantics. There's one global
allocator, allocations come from everywhere, and we want *all* of
them (from tokio threads, OS-level threads, the allocator inside a
library's background thread) to land in the same trace. A single
captured handle achieves that.

Incidental benefit: skipping the `current()` TLS lookup saves a few
ns per sampled alloc. Not the reason, but nice.

### `MemoryProfiler::install()` flow

```rust
static ACTIVE: OnceLock<MemoryProfilerState> = OnceLock::new();

impl MemoryProfiler {
    pub fn install(
        self,
        handle: TelemetryHandle,
    ) -> Result<MemoryProfilerGuard, InstallError> {
        // 1. Install SIGSEGV handler, get Unwinder.
        let unwinder = Unwinder::install().map_err(InstallError::Unwinder)?;

        // 2. Build the producer-side state. The `RingBuffer` (an
        //    `ArrayQueue<RawSample>`) lives here and is also handed
        //    to the flush thread via the `Source` registration in
        //    step 5.
        let ring = std::sync::Arc::new(RingBuffer::new(self.config.ring_capacity));
        let state = MemoryProfilerState {
            unwinder,
            handle: handle.clone(),
            config: self.config.clone(),
            ring: ring.clone(),
        };

        // 3. Publish exactly once. `OnceLock::set` returns `Err(state)`
        //    on a second call, which maps directly to AlreadyInstalled.
        ACTIVE
            .set(state)
            .map_err(|_| InstallError::AlreadyInstalled)?;

        // 4. Register the ring as a `Source` with the existing flush
        //    thread (the same thread that already drains CpuProfiler /
        //    SchedProfiler — see §6). The flush thread's own
        //    allocations (encoder grow, hashmap insert, etc.) re-enter
        //    the hook just like any other thread's; the geometric
        //    sample rate keeps self-pollution to ~0.01% of trace
        //    events. See §7.
        handle.register_source(MemoryProfileSource { ring, liveset, ... });

        Ok(MemoryProfilerGuard { _private: () })
    }
}

#[derive(Debug)]
pub enum InstallError {
    AlreadyInstalled,
    Unwinder(std::io::Error),
}

/// Dropping this guard does NOT uninstall the profiler. The static
/// state lives until process exit. The guard exists to make the
/// API familiar (RAII) and to hold a lifetime if we add a pause/
/// resume later.
pub struct MemoryProfilerGuard { _private: () }
```

The state lives in a `OnceLock<MemoryProfilerState>` and is never
reclaimed because:

1. In-flight hook calls may be reading `ACTIVE.get().unwrap().handle`
   at any moment on any thread. Freeing while a reader holds the
   reference is unsound. We'd need hazard pointers or RCU to free
   safely.
2. The only time we'd want to free is at process exit, when the OS
   reclaims everything anyway.

Cost is ~100 bytes plus the liveset size. Acceptable.

**Why `OnceLock` instead of a raw `AtomicPtr`.** The semantics we
need are exactly "write once, read many, never reclaim," which is
what `OnceLock` encodes in the type system — no `unsafe`, no
pointer-provenance footguns, and the no-uninstall invariant is
enforced by the absence of a safe `take()`-equivalent rather than
by a comment. Hot-path cost is identical: `OnceLock::get()` on a
set value is an `Acquire` load of an internal pointer slot plus a
null check, the same thing the hand-rolled `AtomicPtr::load` would
compile to.

### Graceful shutdown

The flush thread drains the ring every 5 ms. On graceful shutdown
(`drop(guard)` on the dial9 `TelemetryGuard`), the flush thread runs
one final cycle and joins. Any samples pushed in the last 5 ms before
that final cycle are drained; samples pushed *after* the final cycle
but *before* the flush thread observes the stop signal are lost.

For applications that need the absolute last batch of memory-profile
events on disk (e.g. running this in a script that exits immediately
after the work completes), call `handle.flush_memory_profile()` before
dropping the guard:

```rust
guard.handle().flush_memory_profile()?;  // synchronous: pop ring,
                                         // emit events, flush encoder,
                                         // wait for collector batch
                                         // to land in the writer.
guard.graceful_shutdown(Duration::from_secs(5))?;
```

This is a thin wrapper around `MemoryProfileSource::flush(&ctx)` that
runs on the calling thread. The drain's own allocations (encoder
grow, hashmap insert) flow through the hook normally — same as any
other thread's allocations — and the geometric sample rate keeps
self-pollution bounded.

For `kill -9` and similar abrupt termination, the last 5 ms of samples
are lost — same trade-off as every other dial9 source.

### Hot path

```rust
fn alloc(&self, layout: Layout) -> *mut u8 {
    // SAFETY: forwarding to the inner allocator.
    let ptr = unsafe { self.0.alloc(layout) };
    if !ptr.is_null() {
        if let Some(state) = ACTIVE.get() {
            hook::on_alloc(state, ptr, layout.size());
        }
    }
    ptr
}
```

Inside `hook::on_alloc`, after the sampling decision and stack
capture (the hook runs on every thread including the flush thread —
see §7 for why that's fine):

```rust
fn push_sample(
    state: &MemoryProfilerState,
    size: usize,
    addr: *mut u8,
    stack: &[u64],
) {
    let mut sample = RawAlloc::empty();
    sample.tid = current_tid();
    sample.size = size as u64;
    sample.addr = addr as u64;
    sample.ts_ns = state.config.timestamp_mode.stamp(); // see §3
    sample.frame_count = stack.len() as u8;
    sample.frames[..stack.len()].copy_from_slice(stack);

    if state.ring.push(RawSample::Alloc(sample)).is_err() {
        state.dropped_samples.fetch_add(1, Ordering::Relaxed);
    }
}
```

No `TelemetryHandle::current()`, no `record_event`, no encoder
access. `state.ring` is the lock-free `ArrayQueue<RawSample>`; the
flush thread (consolidator) is what eventually walks the ring and
calls `record_encodable_event` on each drained sample — see §5.

### What about `disabled` handles?

`install(handle)` accepts any `TelemetryHandle`, including an inert
one. If the user installs with a disabled handle (e.g. because
`Dial9Config::builder().build_or_disabled()` produced a disabled
runtime after an I/O error), the hot path still runs the sampling
decision + stack capture — wasted work. We could short-circuit by
checking `handle.is_enabled()` at install time and skipping the
`ACTIVE.store` if not. Lean: do this. Single check at install time,
keeps the "did the user enable memory profiling?" / "does dial9
actually have a live trace?" conditions aligned.

### `MemoryProfilingConfig`

```rust
#[derive(Debug, Clone)]
pub struct MemoryProfilingConfig {
    sample_rate_bytes: u64,             // default 512 KiB
    track_liveset: bool,                // default false (off)
    timestamp_mode: TimestampMode,      // default ReusePollStart
    max_liveset_entries: Option<usize>, // default None (unbounded)
    rng_seed: Option<u64>,              // test-only; seeds per-thread RNGs deterministically
    ring_capacity: usize,               // default 4096 slots (≈2.1 MiB)
}

/// How `AllocEvent.timestamp_ns` is populated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TimestampMode {
    /// Reuse the timestamp stamped into TLS by the most recent
    /// `PollStart` on this thread when one is available
    /// (~2 ns TLS load). On threads that have no recorded `PollStart`
    /// yet — blocking-pool workers, user OS threads, library-spawned
    /// threads, or a tokio worker pre-first-poll — fall back to
    /// `clock_monotonic_ns()` (~25 ns vDSO).
    ///
    /// All allocations within the same poll get the same timestamp;
    /// ordering within a poll is lost but analysis still has a usable
    /// time for every event.
    ///
    /// Default.
    ReusePollStart,

    /// Emit events with `timestamp_ns = 0`. Smallest on-disk size
    /// (LEB128-encoded zero is 1 byte). Analysis toolkit can still
    /// group by stack and size; viewer loses time-range filtering and
    /// cross-event correlation.
    ///
    /// Useful when you only care about aggregate hot stacks and want
    /// minimal overhead + trace size.
    None,

    /// Call `clock_monotonic_ns()` per sampled allocation (~25 ns via
    /// vDSO on Linux). Only sampled allocations pay this; unsampled
    /// ones still take the fast path.
    ///
    /// Use this when investigating tight allocation loops or timing
    /// correlation with other events within a single poll.
    Precise,
}

impl MemoryProfiler {
    pub fn builder() -> MemoryProfilerBuilder { ... }
}
```

Built via `#[bon::builder]` as with other dial9 configs.

### Why an enum over a bool for timestamps

Three real modes, not two:
- `ReusePollStart` is the right default — cheap, good enough for
  flamegraph and per-poll analysis.
- `Precise` for tight-loop investigations.
- `None` is actually useful: minimum overhead + minimum trace size
  for customers who just want aggregate stack hotness.

An enum also leaves room to add modes without breaking existing calls
(e.g. `Coarse` at 10ms resolution if profiling shows vDSO overhead
under extreme load). `#[non_exhaustive]` protects the semver.

---

## 9. Liveset tracking

When `track_liveset = true`, the **consolidator thread** maintains:

```rust
struct LivesetEntry {
    /// Copied onto `FreeEvent.size` on dealloc.
    size: u64,
    /// Copied onto `FreeEvent.alloc_timestamp_ns` on dealloc. This is
    /// the `AllocEvent.timestamp_ns` we emitted for the matching alloc.
    timestamp_ns: u64,
    // No stack frames here. The per-live-alloc memory cost of storing
    // a full call stack (64 × 8 B) dominates total liveset overhead.
    // Stack-attributed leak analysis joins FreeEvent→AllocEvent by
    // `addr` inside a single retained segment instead.
}

// Single-threaded — only the consolidator reads or writes it.
// Plain HashMap; no synchronization, no concurrent-data-structure
// gymnastics. The consolidator's own allocations during HashMap
// rehash do re-enter the hook, but the ring push is allocator-quiet
// so the recursion is bounded by the geometric sample rate (§7).
liveset: HashMap<usize, LivesetEntry>,
```

### Why a plain `HashMap` works now

The earlier draft of this design proposed `crossbeam-skiplist::SkipMap`
because the original (precise) design needed a concurrent map: every
thread that allocates would `insert(addr, entry)`, every thread that
deallocs would `remove(addr)`. That's many writers, many readers,
running inside the allocator hook — `Mutex<HashMap>` deadlocks,
`DashMap`'s shard locks risk priority-inversion, etc.

The ring design changes the access pattern entirely:

- Producer threads push `RawSample::Alloc { addr, ... }` and
  `RawSample::Free { addr }` records into the ring. They never touch
  the liveset directly.
- The consolidator pops records, calls `liveset.insert(addr, entry)`
  or `liveset.remove(&addr)` from a single thread.
- No locks. No CAS loops. No epoch reclamation. No `catch_unwind`
  around `papaya` calls. The consolidator's own allocations during
  `HashMap` rehash *do* re-enter the hook, but the ring push is
  allocator-quiet so the recursion can't cascade — see §7 for the
  bounded-self-sampling argument.

`std::collections::HashMap` is fine. Insert / remove on a 16K-entry
map costs ~20–40 ns warm. We pay that on the consolidator, not on
the allocating thread.

If the per-pop cost shows up in profiling we can swap to
`hashbrown::HashMap` (already a transitive dep via `dial9-trace-format`)
or pre-size the map. None of those are blocking decisions.

### Bloom filter optimization

Not needed. Every dealloc gets pushed to the ring; the consolidator
does the lookup; the lookup misses for unsampled addresses; the miss
is a single `HashMap` get (~10–20 ns) on the consolidator side. We
can add a bloom filter in front later if the consolidator's drain
loop shows up in profiles, but at the realistic 100K-records/sec ring
load that's ~2 ms/sec of consolidator work — already invisible.

### Bounded liveset

Unbounded liveset can consume significant memory in a long-running
leaky process (which is, ironically, when you want it on).
`max_liveset_entries` caps the total count; when full, the overflow
policy (now applied on the consolidator) is:

- New `RawSample::Alloc`: still emit `AllocEvent`, but skip liveset
  insert. Emit a rate-limited warning event (once per 60s).
- `RawSample::Free`: still lookup (miss on unsampled, or miss on
  skipped). No-op on miss.

Traces stay useful for hotspot analysis even if the liveset overflows.
Leak analysis loses fidelity in that range.

Default: `None` (unbounded). Users opt in to a cap.

### Liveset memory cost

The consolidator's `HashMap<usize, LivesetEntry>` grows with the number
of live sampled allocations:

- `LivesetEntry` is 16 bytes (`size: u64`, `timestamp_ns: u64`).
- HashMap overhead is ~16 bytes/entry (key + slot metadata).
- **Total: ~32 bytes per live sampled allocation.**

At the default 512 KiB sample rate and a 1 GiB live heap, expect
~2K live entries → ~64 KiB. At a 10 GiB live heap, ~20K entries →
~640 KiB. For a leaky service that grows to 100K live sampled
allocations, the liveset costs ~3.2 MiB on the consolidator thread.

This is on top of the §5 ring buffer (~4.2 MiB at default capacity).

### Liveset overhead

The ring design moves liveset cost off the allocating thread, but the
free-queue push still costs the producer something. Measured numbers
(see `mem-profile-experiments/benches/dealloc_overhead.rs`):

- **Per sampled alloc on the producer:** nothing extra. The ring push
  carries the `addr`; the consolidator does the insert.
- **Per dealloc on the producer (liveset on):** one
  `free_queue.push(RawFree)` — **9.3 ns uncontended**, ~3 µs under
  pathological 8-thread contention with all threads pushing flat-out.
- **Per sample on the consolidator:** one `HashMap` insert or remove
  (~20–40 ns warm).

The 9.3 ns figure is roughly half of what the WIP/precise design
costs for a SkipMap lookup (~30–60 ns). It's also consistent with
typical `crossbeam_queue::ArrayQueue::push` benchmarks for small
24-byte records.

**Sizing the free queue.** Because dealloc rates are much higher
than sampled-alloc rates (frees are pushed unconditionally when
liveset is on, allocs only when sampled), we make the free queue
**8× larger** than the alloc queue. With default capacity that's
32K slots × 24 B = 768 KiB. Same lock-free `ArrayQueue`, just sized
for the higher throughput.

**Throughput ceiling.** The `dealloc_overhead` bench shows the
8-thread MPMC `ArrayQueue` saturates at ~2.7 M pushes/sec aggregate
under flat-out contention. A service doing 1 GB/s alloc at 64-byte
average objects → ~15 M deallocs/sec aggregate, which is well above
this ceiling. **At those rates we need a producer-side optimization**;
two options that don't require leaving the ring-buffer architecture:

1. **Per-thread free buffer.** Each producer batches frees in a TLS
   `[RawFree; 64]` array; flush to the free queue as one batched
   push every 64 frees. Converts 64 contended pushes into 1, dropping
   contention 64×.
2. **Producer-side bloom filter.** Maintain a process-global bloom
   filter of "addresses we sampled at alloc time." Producers test the
   filter (one hash + bit-check, ~5 ns) before pushing; misses (99.9%
   of deallocs) skip the push entirely. Adds a small false-positive
   rate (extra free-queue pushes that the consolidator drops) but
   massively reduces traffic.

Option 1 is conceptually simpler; option 2 is more invasive but has
larger payoff at extreme rates. Pick at implementation time based on
whether the dealloc bench shows actual production contention.

The liveset is still off by default because even 9.3 ns per dealloc
on a dealloc-heavy workload adds up (~5% overhead at 5M deallocs/sec
single-threaded), and the producer-side optimization above is opt-in
work.



---

## 10. Overhead budget

Target: <1% at default settings (512 KiB sample rate, no liveset).

**Context: typical allocator latencies.** For reference, a single
`malloc`/`free` call on modern allocators (glibc, jemalloc, mimalloc)
takes ~20-80ns uncontended. Under contention or with fragmentation,
individual calls can spike to 1-10µs. Our fast-path overhead (~5ns)
is well within the noise of a single allocation; the sampled-path
overhead (~300-500ns) is comparable to a single contended malloc.

Per-allocation fast path (unsampled, ~99.9% of calls):
- 1 `OnceLock::get()` (Acquire load + null check, ~1 ns)
- 1 subtract + compare on per-thread `next_sample_bytes` (~1 ns)
- Return

Measured **~3.57 ns** total in the precise/inline (WIP) design and
**~1.10 ns** in the ring design — see §5 for the full bench table.
The ring design is faster because it doesn't take a per-call
`ReentrancyGuard`: the ring push is allocator-quiet by construction,
so the unsampled path is just the sampling counter check and a
return.

For a service doing 1M allocs/sec: ~1–4 ms/sec of CPU per core, i.e.
0.1–0.4% of one core. Well under the <1% budget. Most services do
far fewer.

Per-sampled allocation (~0.1% of calls at 512 KiB sample rate):
- Stack capture (~110 ns for 20 frames warm-cache; ~200–400 ns cold).
- Build `RawAlloc` (~1 KiB) on the stack (~30 ns — copying frames).
- `ArrayQueue::push` of the 1 KiB record (~50–100 ns uncontended;
  more under contention).
- Optional timestamp call (~25 ns when `TimestampMode::Precise`).

Measured **~1226 ns** for `ring_sampled_deep20` (depth-20 stack +
RawAlloc assemble + push, with `MAX_FRAMES = 128`). This is dominated
by stack capture and the 1 KiB record copy; the rest of the path
adds ~100 ns.

At 2K samples/sec: ~2.5 ms/sec of CPU per core (~0.25% of one core),
well under budget.

**Dealloc overhead with liveset:** ~9.3 ns per dealloc uncontended
(one `free_queue.push(RawFree)` of a 24-byte record). Under heavy
8-thread MPMC contention the queue saturates near 2.7 M pushes/sec
aggregate; for services with 10M+ deallocs/sec we'd add a per-thread
free-buffer batch (see §9 "Throughput ceiling"). Default is
liveset-off, so most users never pay this.

### Benchmarking

The throwaway crate `mem-profile-experiments/` (excluded from the
workspace, contains both design implementations) is what generated
the numbers above. It's a self-contained bench bed:
`cargo bench --bench hook_overhead` measures unsampled / mixed paths,
`cargo bench --bench sampled_path` measures the sampled path, and
`cargo bench --bench consolidator_throughput` exercises the
consolidator-side cost. The crate is meant to be kept until the
ring design is shipped, then deleted.

**We still need an integration benchmark** that exercises the full
`Dial9Allocator → ring → consolidator → trace` pipeline before we
can claim end-to-end overhead numbers. Shape:

- High-frequency small allocations (tight `Box::new(T)` or
  `Vec::with_capacity(small)` loops) to measure fast-path cost.
- Realloc growth (e.g. `Vec::push` in a loop) to measure the
  free+alloc-sampled path.
- Mixed sizes across the sample-rate boundary so unbiasing has
  variance to work with.
- Comparison matrix: (no profiler) vs (sampling only) vs (sampling +
  liveset).

Lives alongside the hook implementation in the same PR. Likely
`benches/memory_profiling_bench.rs` with per-case Criterion groups.

---

## 11. Viewer changes

### Allocation flamegraph

Same UX as CPU flamegraph — click a poll, or shift-drag a time range,
see a flamegraph. The sample value is
`AllocEvent.size * weight_correction(AllocEvent.size, sample_rate)`,
where `sample_rate` is read once per segment from `SegmentMetadata`.
Code lives alongside the existing flamegraph, driven by a new
aggregator that consumes `AllocEvent` instead of `CpuSample`.

### Per-task allocation chart

For each task, total bytes allocated + sampled count. Sort by size;
shows leaky / hot tasks at a glance. Read directly from trace:

```
sample_rate = segment_metadata["memory_profile.sample_rate_bytes"]
for each AllocEvent e:
    worker = worker_for(e.tid, e.timestamp_ns)  // via WorkerUnpark history
    poll = poll_containing(e.timestamp_ns, worker)
    task = poll.task_id
    agg[task] += e.size * weight_correction(e.size, sample_rate)
```

This is the same join the viewer already does between `CpuSample` and
polls. Allocations with no containing poll go into an "unassociated"
bucket — common for background threads, first allocations on new
workers, etc.

### Leak view (liveset only)

Show allocations with no matching free at end-of-trace:
```
live = {}
for each AllocEvent: live[addr] = (size, stack)
for each FreeEvent:  live.remove(addr)
# group by stack, sort by total bytes
```

The `(size, stack)` lookup is cheap: one hashmap entry per live
sampled allocation in the trace, joined as we stream events.

---

## 12. Analysis toolkit

Extend `dial9-viewer/skills/analyze.js` with:

- `allocationsByTask()` — groups by task, weighted.
- `topAllocationStacks(n)` — flamegraph-style stack aggregation,
  unbiased.
- `leakCandidates(minBytes)` — live allocations grouped by stack,
  above a threshold. Only meaningful when the trace has `FreeEvent`s.

CLI:

```bash
cargo run --example analyze_trace --features analysis -- \
    --memory trace.0.bin.gz
```

Emits: per-task alloc totals, top 20 stacks, leak candidates if
available.

New skill doc: `dial9-viewer/skills/memory.md` covering recipes for
common questions ("what allocated the most bytes in this time range?",
"which stacks show the largest retained heap?", etc.).

---

## 13. Testing strategy

**Install-once is permanent per process.** `MemoryProfiler::install`
publishes a process-global `OnceLock<MemoryProfilerState>` that is
never reclaimed (see §8 for why). Tests that exercise different
memory-profiling configurations therefore each need their own
process.

We do **not** add a test-only `reset_for_testing()` escape hatch:

- The soundness reason for no-reclaim (hooks on any thread may be
  reading `ACTIVE.get()` at any moment) applies in tests too. A
  `reset_for_testing()` that doesn't implement hazard pointers is a
  data race; one that does has cost we don't want to pay in prod
  builds.
- `cargo nextest run` already runs each `#[test]` in its own process
  by default, so integration-style tests organized as
  `tests/memory_profiling_*.rs` files pick up a fresh state
  automatically. This is the approach the existing S3 worker tests
  use (see `tests/s3_integration.rs`).
- Contributors who run tests via `cargo test` (which reuses the
  process across tests within a binary) need to structure tests
  within a single file so they share a consistent profiler
  configuration, or move into separate `tests/` files. This is
  called out in the tests' module docs.

### Test categories

1. **Unit tests** around the sampling math (in-crate, do not
   `install()` a profiler):
   - Empirical sampling rate matches target rate within ±10% over
     ≥ 10k simulated allocs.
   - `draw_exponential` distribution sanity (mean, variance).
   - Deterministic via `rng_seed`.
2. **Integration** with a dial9 runtime (each in its own
   `tests/memory_profiling_*.rs` file, installing the global
   allocator per file):
   - Alloc a known pattern, inspect the trace for expected events and
     approximate counts.
   - Verify `AllocEvent.tid` matches the thread the alloc ran on, and
     that joining against `WorkerUnpark` history recovers the correct
     worker when the thread is a tokio worker.
   - Verify FP unwinder produces stacks whose top frame matches the
     allocation callsite.
3. **Liveset** round-trip (own test file):
   - Alloc N things, free M, verify liveset.len() == N - M.
   - Verify `FreeEvent` count matches sampled-alloc count of freed
     pointers.
   - Verify `FreeEvent.size` and `FreeEvent.alloc_timestamp_ns` match
     the original `AllocEvent` values (the rotation-robustness
     contract from §3).
4. **Reentrancy** (own test file):
   - Record a tracing event whose subscriber allocates; confirm no
     infinite recursion and the outer event still reaches the trace.
5. **Realloc** (own test file): alloc, realloc to larger (in-place)
   and larger (moved), verify free-of-old + alloc-of-new are emitted
   per jemalloc rules.
6. **Rotation robustness** (own test file): allocate a long-lived
   buffer, trigger rotation so the `AllocEvent` is evicted, free the
   buffer, verify `FreeEvent.size` is non-zero and analysis can still
   compute the net-bytes delta.
7. **Concurrency (shuttle)**: Use [shuttle](https://github.com/awslabs/shuttle)
   to test the reentrancy guard and liveset under simulated thread
   interleavings. Key scenarios: two threads sampling simultaneously,
   a thread freeing while another inserts, and epoch advancement under
   contention. Shuttle's deterministic scheduler can surface races that
   stress tests miss.

---

## 14. Open questions

1. **Do we want per-alloc allocator latency?** Wrapping `self.0.alloc`
   with a `clock_monotonic_ns` pair gives allocator latency, which is
   interesting for jemalloc/mimalloc fragmentation investigations.
   Cost: +50ns per sampled alloc (two vDSO calls). Easy to add as a
   field later. If added, this should use a lower sample rate than the
   default 512 KiB (e.g. 4 MiB) since the latency measurement adds
   overhead to every sampled alloc and the signal-to-noise ratio is
   good even at lower rates.

2. **MUSL / static builds.** The `safe_load` trampoline + SIGSEGV chain
   works on glibc. Need to verify on musl (Alpine containers). If it
   doesn't work reliably, **conditionally compile out** the frame-pointer
   unwinder on musl targets (`#[cfg(not(target_env = "musl"))]`) and
   fall back to no-stack-capture mode (events still emitted with empty
   stacks). We should not ship untested code paths — if musl isn't
   validated before release, it's compiled out.

3. **Multi-runtime selection.** With a single captured handle, all
   allocations land in the trace owned by whichever runtime the user
   passed to `install()`. If a service runs multiple dial9 runtimes
   and wants per-runtime allocation attribution, we'd need a
   different strategy (e.g. a handle lookup via a trait the allocator
   could call, or tagging each TLS buffer with a runtime-id). I don't
   think this use case is real yet; punt until it is.

4. **Dynamic ring-buffer sizing.** Today `ring_capacity` is fixed at
   install time (default 4096 slots). If a service hits sustained
   sample bursts that exhaust the ring, the producer side starts
   dropping samples. We could (a) leave it on the user to bump
   `ring_capacity` after seeing dropped-samples > 0, (b) auto-resize
   the ring on the consolidator thread when `dropped_samples` crosses
   a threshold, or (c) fall back to a per-thread overflow buffer
   that the consolidator drains when the main ring drains. Punt
   until the dropped-samples counter shows it's actually a problem
   in practice — at default sample rates the ring should hold
   minutes of samples.

---

## 15. Rollout

1. **`Unwinder::install()` / `capture()` public in `perf-self-profile`.**
   Already shipped — see d0b0380 / #396.

2. **Add `tid` to `WorkerParkEvent` and `WorkerUnparkEvent`.**
   Standalone trace-format PR, additive (does not break old parsers).
   Producer side calls `current_tid()` at park/unpark time (~5 ns
   syscall on Linux, ~1 ns TLS lookup elsewhere). Lets analysis
   tools resolve `tid → worker_id` from the trace alone, without
   runtime cooperation. Decouples future event sources (memory
   profiler in this design, possibly an `mmap` source later) from
   tokio-runtime state. Bonus: surfaces `block_in_place` cleanly
   — a single `worker_id` backed by different `tid`s over its
   lifetime is now visible in the trace. Trace format is additive,
   but per AGENTS.md a format change requires regenerating the
   demo trace. Tests: park/unpark round-trip, blocking-pool
   thread spawn shows up with its own tid range.

3. **Extract the `Source` trait (§6) and migrate `CpuProfiler` /
   `SchedProfiler` to it.** Standalone PR, no behavior change. Tests
   pass through the trait. This unblocks the memory profiler from
   landing without retrofitting.

4. **Extract sampling primitives from `task_dumped.rs`.** Move
   `SplitMix64` and `draw_exponential` into a new
   `dial9-tokio-telemetry/src/sampling.rs` module
   (`pub(crate)`). Update the existing task-dump code path to import
   from there. Pure refactor — no behavior change. Unblocks the
   memory profiler reusing the same sampling primitives without
   inventing a parallel implementation. See §1 for the API.

5. **Ring-buffer scaffolding: `RawSample`, `RingBuffer`, register the
   `Source` with the existing flush thread.** Plumb the ring through
   the recorder so `MemoryProfiler::install` registers the
   `MemoryProfileSource` and the existing flush loop drains it every
   5 ms. No allocator hook yet — exercise the ring with a synthetic
   producer in tests.

6. **`Dial9Allocator<A>` + sampling hook + ring push, no liveset.**
   Gate behind a `memory-profiling` feature flag.
   `MemoryProfiler::builder().install(guard.handle())` publishes the
   captured handle via `OnceLock<MemoryProfilerState>`. Tests:
   overhead, approximate sampling rate, stacks symbolize correctly,
   allocations on non-tokio threads are captured, second `install()`
   call returns `AlreadyInstalled`, dial9-internal allocations show
   up in the trace as a small fraction (~0.01% at default rate) per
   §7.

7. **Liveset tracking + `FreeEvent`s.** Add `RawSample::Free` push
   path and consolidator-side `HashMap` lookup. Tests per §13. This
   is when `track_liveset(true)` starts emitting `FreeEvent`s.

8. **Realloc handling per jemalloc rules.** Dedicated tests for the
   four cases.

9. **Allocation-focused integration benchmark in CI.** Land alongside
   step 6 (sampling) and extend in step 7 (liveset). Fail CI on
   regression. See §10 "Benchmarking" — needs to exercise high-
   frequency small allocs, realloc growth, and mixed sizes. The
   throwaway `mem-profile-experiments/` benches stay out of CI; the
   integration bench takes their place.

10. **Viewer: alloc flamegraph.** Parser + UI.

11. **Viewer: per-task chart + leak view.** More UI.

12. **Analysis toolkit + skill docs.**

Each step ships independently. Step 2 is a small standalone trace-
format change that can land in parallel with steps 3 and 4 (both
pure refactors). Steps 3–6 are the blocking chain for the memory
profiler; 7+ can interleave or run in parallel.
