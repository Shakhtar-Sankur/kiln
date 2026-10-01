//! A persistent thread pool for data-parallel kernels (from ferrolm). `run(n, f)` calls
//! `f(0..n)` across the workers and the calling thread and returns when all
//! calls are done. Workers spin briefly between jobs, because a decode step
//! issues hundreds of small jobs and waking a sleeping thread costs more
//! than many of them take.
//!
//! Tasks are handed out dynamically, but each task writes a fixed part of
//! the output, so results never depend on the number of threads or on which
//! thread ran what.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

type Job = *const (dyn Fn(usize) + Sync);

struct Shared {
    generation: AtomicU64,
    job: UnsafeCell<Option<Job>>,
    tasks: AtomicUsize,
    next: AtomicUsize,
    /// Workers that have not yet finished the current generation.
    pending: AtomicUsize,
    shutdown: AtomicBool,
    lock: Mutex<()>,
    wake: Condvar,
}

// The job pointer is written only while no worker reads it (between
// generations) and the closure it points to is `Sync`.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

const SPINS: u32 = 1 << 16;

impl Pool {
    /// A pool that runs jobs on `threads` threads in total, including the
    /// caller.
    pub fn new(threads: usize) -> Pool {
        let shared = Arc::new(Shared {
            generation: AtomicU64::new(0),
            job: UnsafeCell::new(None),
            tasks: AtomicUsize::new(0),
            next: AtomicUsize::new(0),
            pending: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            lock: Mutex::new(()),
            wake: Condvar::new(),
        });
        let workers = (1..threads.max(1))
            .map(|_| {
                let s = Arc::clone(&shared);
                // Generated kernels keep packing panels and row scratch on
                // the stack.
                std::thread::Builder::new()
                    .stack_size(64 << 20)
                    .spawn(move || worker(&s))
                    .expect("spawn worker")
            })
            .collect();
        Pool { shared, workers }
    }

    /// One thread per available core.
    pub fn with_all_cores() -> Pool {
        Pool::new(cores())
    }

    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    pub fn run(&self, tasks: usize, f: &(dyn Fn(usize) + Sync)) {
        if tasks == 0 {
            return;
        }
        if tasks == 1 || self.workers.is_empty() {
            (0..tasks).for_each(f);
            return;
        }
        let s = &*self.shared;
        // SAFETY: no worker reads `job` until `generation` changes below, and
        // this call does not return until every worker has finished with it,
        // so erasing the closure's lifetime is sound.
        unsafe {
            let job: Job = std::mem::transmute::<&(dyn Fn(usize) + Sync), Job>(f);
            *s.job.get() = Some(job);
        }
        s.tasks.store(tasks, Ordering::Relaxed);
        s.next.store(0, Ordering::Relaxed);
        s.pending.store(self.workers.len(), Ordering::Relaxed);
        s.generation.fetch_add(1, Ordering::Release);
        drop(s.lock.lock().unwrap());
        s.wake.notify_all();
        drain(s, f);
        let mut spins = 0u32;
        while s.pending.load(Ordering::Acquire) != 0 {
            spins += 1;
            if spins < SPINS {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
}

fn drain(s: &Shared, f: &(dyn Fn(usize) + Sync)) {
    let n = s.tasks.load(Ordering::Relaxed);
    loop {
        let i = s.next.fetch_add(1, Ordering::Relaxed);
        if i >= n {
            break;
        }
        f(i);
    }
}

fn worker(s: &Shared) {
    let mut seen = 0u64;
    loop {
        let mut spins = 0u32;
        loop {
            if s.generation.load(Ordering::Acquire) != seen || s.shutdown.load(Ordering::Relaxed) {
                break;
            }
            spins += 1;
            if spins < SPINS {
                std::hint::spin_loop();
                continue;
            }
            let g = s.lock.lock().unwrap();
            let _g = s
                .wake
                .wait_while(g, |_| {
                    s.generation.load(Ordering::Acquire) == seen
                        && !s.shutdown.load(Ordering::Relaxed)
                })
                .unwrap();
            break;
        }
        if s.shutdown.load(Ordering::Relaxed) {
            return;
        }
        seen = s.generation.load(Ordering::Acquire);
        // SAFETY: set before `generation` was published and kept alive until
        // `pending` reaches zero.
        let job = unsafe { (*s.job.get()).expect("job set") };
        drain(s, unsafe { &*job });
        s.pending.fetch_sub(1, Ordering::Release);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Relaxed);
        drop(self.shared.lock.lock().unwrap());
        self.shared.wake.notify_all();
        for w in self.workers.drain(..) {
            w.join().ok();
        }
    }
}

/// A raw pointer that tasks use to write disjoint parts of one output.
#[derive(Clone, Copy)]
pub struct Out<T>(*mut T);

unsafe impl<T> Send for Out<T> {}
unsafe impl<T> Sync for Out<T> {}

impl<T> Out<T> {
    pub fn new(s: &mut [T]) -> Out<T> {
        Out(s.as_mut_ptr())
    }

    /// # Safety
    /// Concurrent callers must use disjoint ranges within the original slice.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice(&self, start: usize, len: usize) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(start), len) }
    }

    /// # Safety
    /// `i` is in bounds and no other task writes it concurrently.
    pub unsafe fn read(&self, i: usize) -> T
    where
        T: Copy,
    {
        unsafe { self.0.add(i).read() }
    }

    /// # Safety
    /// As for `write`, for every element the caller touches through it.
    pub unsafe fn ptr(&self, i: usize) -> *mut T {
        unsafe { self.0.add(i) }
    }

    /// # Safety
    /// `i` is in bounds and no other task writes it concurrently.
    pub unsafe fn write(&self, i: usize, v: T) {
        unsafe { self.0.add(i).write(v) }
    }
}

/// The number of cores this process may use.
pub fn cores() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_every_task_exactly_once() {
        let pool = Pool::new(4);
        for n in [1usize, 2, 7, 100, 1000] {
            let mut v = vec![0u32; n];
            let out = Out::new(&mut v);
            pool.run(n, &|i| unsafe { out.slice(i, 1)[0] += 1 });
            assert!(v.iter().all(|&x| x == 1), "n={n}");
        }
        // Many small jobs in a row, as in a decode step.
        let hits = AtomicUsize::new(0);
        for _ in 0..2000 {
            pool.run(8, &|_| {
                hits.fetch_add(1, Ordering::Relaxed);
            });
        }
        assert_eq!(hits.into_inner(), 16000);
    }
}
