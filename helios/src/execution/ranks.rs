//! Direct scheduler execution on one CUDA rank or a coordinated rank group.

use super::{
    ExecutionManifest, FrameCompletion, GraphExecutor, RankStream, TensorParallelExecutor, Upload,
};
use crate::{Result, Shutdown, cache::Fence};

pub enum RankPeers {
    Processes(Vec<RankStream>),
    Threads(Vec<ExecutionManifest>),
}

pub(crate) enum RankExecutor {
    Threaded(Box<super::threaded::ThreadedExecutor>),
    Local(Box<GraphExecutor>),
    Parallel(Box<TensorParallelExecutor>),
}

impl RankExecutor {
    /// # Safety
    /// Prepared graphs, contexts, buffers, and b12x IPC allocations must remain
    /// retained until coordinated drain. Peer sockets identify ranks 1..world.
    pub(crate) unsafe fn new(
        manifest: ExecutionManifest,
        peers: Option<RankPeers>,
    ) -> Result<Self> {
        match peers {
            Some(RankPeers::Threads(ranks)) => {
                unsafe { super::threaded::ThreadedExecutor::new(manifest, ranks) }
                    .map(|executor| Self::Threaded(Box::new(executor)))
            }
            Some(RankPeers::Processes(peers)) => {
                unsafe { TensorParallelExecutor::new(manifest, peers) }
                    .map(|executor| Self::Parallel(Box::new(executor)))
            }
            None => unsafe { GraphExecutor::new(manifest) }
                .map(|executor| Self::Local(Box::new(executor))),
        }
    }

    pub(crate) fn submit(
        &mut self,
        graph: usize,
        uploads: &[Upload<'_>],
        grammar: bool,
    ) -> Result<Option<Fence>> {
        match self {
            Self::Local(executor) => executor.submit(graph, uploads, grammar),
            Self::Parallel(executor) => executor.submit(graph, uploads, grammar),
            Self::Threaded(executor) => executor.submit(graph, uploads, grammar),
        }
    }

    pub(crate) fn poll(&mut self) -> Result<Option<FrameCompletion>> {
        match self {
            Self::Local(executor) => executor.poll(),
            Self::Parallel(executor) => executor.poll(),
            Self::Threaded(executor) => executor.poll(),
        }
    }

    pub(crate) fn outstanding(&self) -> usize {
        match self {
            Self::Local(executor) => executor.outstanding(),
            Self::Parallel(executor) => executor.outstanding(),
            Self::Threaded(executor) => executor.outstanding(),
        }
    }

    /// Copies one prepared boundary buffer to pinned host storage. Pipeline
    /// stages execute as local graphs; tensor-parallel and threaded executors
    /// reject boundary reads rather than silently returning another rank's data.
    pub(crate) fn read_buffer(
        &mut self,
        buffer: &str,
        host: &crate::cuda::PinnedHost,
    ) -> Result<()> {
        match self {
            Self::Local(executor) => executor.read_buffer(buffer, host),
            Self::Parallel(_) | Self::Threaded(_) => {
                Err("pipeline boundary reads require local stage execution".into())
            }
        }
    }

    /// Copies a prepared boundary buffer range to pinned host storage.
    pub(crate) fn read_buffer_range(
        &mut self,
        buffer: &str,
        host: &crate::cuda::PinnedHost,
        device_offset: usize,
        nbytes: usize,
    ) -> Result<()> {
        match self {
            Self::Local(executor) => {
                executor.read_buffer_range(buffer, host, device_offset, nbytes)
            }
            Self::Parallel(_) | Self::Threaded(_) => {
                Err("pipeline boundary reads require local stage execution".into())
            }
        }
    }

    /// Uploads pinned host storage into one prepared boundary buffer.
    pub(crate) fn write_buffer(
        &mut self,
        buffer: &str,
        host: &crate::cuda::PinnedHost,
    ) -> Result<()> {
        match self {
            Self::Local(executor) => executor.write_buffer(buffer, host),
            Self::Parallel(_) | Self::Threaded(_) => {
                Err("pipeline boundary writes require local stage execution".into())
            }
        }
    }

    /// Allocates pinned host storage for one boundary activation.
    pub(crate) fn host_buffer(&mut self, nbytes: usize) -> Result<crate::cuda::PinnedHost> {
        match self {
            Self::Local(executor) => executor.host_buffer(nbytes),
            Self::Parallel(_) | Self::Threaded(_) => {
                Err("pipeline boundary buffers require local stage execution".into())
            }
        }
    }

    #[cfg(feature = "server")]
    pub(crate) fn manifest(&self) -> &ExecutionManifest {
        match self {
            Self::Local(executor) => executor.manifest(),
            Self::Parallel(executor) => executor.manifest(),
            Self::Threaded(executor) => executor.manifest(),
        }
    }

    #[cfg(feature = "server")]
    pub(crate) fn can_submit(&self, grammar: bool) -> bool {
        match self {
            Self::Local(executor) => executor.can_submit(grammar),
            Self::Parallel(executor) => executor.can_submit(grammar),
            Self::Threaded(executor) => executor.can_submit(grammar),
        }
    }

    #[cfg(feature = "server")]
    pub(crate) fn next_fence(&self) -> Result<Fence> {
        match self {
            Self::Local(executor) => executor.next_fence(),
            Self::Parallel(executor) => executor.next_fence(),
            Self::Threaded(executor) => executor.next_fence(),
        }
    }

    pub(crate) fn close(&mut self) -> Shutdown {
        match self {
            Self::Local(executor) => executor.close(),
            Self::Parallel(executor) => executor.close(),
            Self::Threaded(executor) => executor.close(),
        }
    }
}
