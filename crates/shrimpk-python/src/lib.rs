use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use shrimpk_core::{EchoConfig, MemoryId};
use shrimpk_memory::EchoEngine;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Python wrapper for EchoConfig
#[pyclass]
#[derive(Clone)]
struct PyEchoConfig {
    inner: EchoConfig,
}

#[pymethods]
impl PyEchoConfig {
    #[staticmethod]
    fn auto_detect() -> Self {
        Self {
            inner: EchoConfig::auto_detect(),
        }
    }

    #[staticmethod]
    fn minimal() -> Self {
        Self {
            inner: EchoConfig::minimal(),
        }
    }

    #[staticmethod]
    fn standard() -> Self {
        Self {
            inner: EchoConfig::standard(),
        }
    }

    #[staticmethod]
    fn full() -> Self {
        Self {
            inner: EchoConfig::full(),
        }
    }

    #[getter]
    fn max_memories(&self) -> usize {
        self.inner.max_memories
    }

    #[getter]
    fn similarity_threshold(&self) -> f32 {
        self.inner.similarity_threshold
    }

    #[getter]
    fn max_echo_results(&self) -> usize {
        self.inner.max_echo_results
    }
}

/// Python wrapper for echo results
#[pyclass]
struct PyEchoResult {
    #[pyo3(get)]
    memory_id: String,
    #[pyo3(get)]
    content: String,
    #[pyo3(get)]
    similarity: f32,
    #[pyo3(get)]
    final_score: f64,
    #[pyo3(get)]
    source: String,
    #[pyo3(get)]
    labels: Vec<String>,
}

/// Python wrapper for memory stats
#[pyclass]
struct PyMemoryStats {
    #[pyo3(get)]
    total_memories: usize,
    #[pyo3(get)]
    index_size_bytes: u64,
    #[pyo3(get)]
    ram_usage_bytes: u64,
    #[pyo3(get)]
    avg_echo_latency_ms: f64,
    #[pyo3(get)]
    total_echo_queries: u64,
}

/// The main Echo Memory engine for Python.
///
/// Usage:
///     from shrimpk import EchoMemory, PyEchoConfig
///     mem = EchoMemory()
///     mid = mem.store("Rust is great for systems programming")
///     results = mem.echo("systems programming")
///     for r in results:
///         print(f"{r.content} (score: {r.final_score:.3f})")
#[pyclass]
struct EchoMemory {
    engine: Arc<Mutex<EchoEngine>>,
    runtime: tokio::runtime::Runtime,
}

#[pymethods]
impl EchoMemory {
    #[new]
    fn new(config: Option<PyEchoConfig>) -> PyResult<Self> {
        let config = config
            .map(|c| c.inner)
            .unwrap_or_else(EchoConfig::auto_detect);
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        let engine = EchoEngine::new(config)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to init engine: {}", e)))?;

        Ok(Self {
            engine: Arc::new(Mutex::new(engine)),
            runtime,
        })
    }

    /// Store a memory. Returns the memory ID as a string.
    ///
    /// Args:
    ///     text: The text content to store.
    ///     source: Where this memory came from (default: "python").
    ///
    /// Returns:
    ///     str: The UUID of the stored memory.
    fn store(&self, text: &str, source: Option<&str>) -> PyResult<String> {
        let source = source.unwrap_or("python");
        let engine = self.engine.clone();
        let text = text.to_string();
        let source = source.to_string();

        self.runtime.block_on(async move {
            let engine = engine.lock().await;
            engine
                .store(&text, &source)
                .await
                .map(|id| id.to_string())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Find memories that resonate with the query.
    ///
    /// Args:
    ///     query: The text to find matching memories for.
    ///     max_results: Maximum number of results (default: 10).
    ///
    /// Returns:
    ///     list[PyEchoResult]: Matching memories sorted by score.
    fn echo(&self, query: &str, max_results: Option<usize>) -> PyResult<Vec<PyEchoResult>> {
        let max_results = max_results.unwrap_or(10);
        let engine = self.engine.clone();
        let query = query.to_string();

        self.runtime.block_on(async move {
            let engine = engine.lock().await;
            let results = engine
                .echo(&query, max_results)
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            Ok(results
                .into_iter()
                .map(|r| PyEchoResult {
                    memory_id: r.memory_id.to_string(),
                    content: r.content,
                    similarity: r.similarity,
                    final_score: r.final_score,
                    source: r.source,
                    labels: r.labels,
                })
                .collect())
        })
    }

    /// Forget (remove) a memory by its ID.
    ///
    /// Args:
    ///     memory_id: The UUID string of the memory to remove.
    fn forget(&self, memory_id: &str) -> PyResult<()> {
        let engine = self.engine.clone();
        let id = uuid::Uuid::parse_str(memory_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid UUID: {}", e)))?;
        let memory_id = MemoryId::from_uuid(id);

        self.runtime.block_on(async move {
            let engine = engine.lock().await;
            engine
                .forget(memory_id)
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Get current engine statistics.
    ///
    /// Returns:
    ///     PyMemoryStats: Current memory count, RAM usage, query stats.
    fn stats(&self) -> PyResult<PyMemoryStats> {
        let engine = self.engine.clone();
        self.runtime.block_on(async move {
            let engine = engine.lock().await;
            let s = engine.stats().await;
            Ok(PyMemoryStats {
                total_memories: s.total_memories,
                index_size_bytes: s.index_size_bytes,
                ram_usage_bytes: s.ram_usage_bytes,
                avg_echo_latency_ms: s.avg_echo_latency_ms,
                total_echo_queries: s.total_echo_queries,
            })
        })
    }

    /// Persist the memory store to disk.
    fn persist(&self) -> PyResult<()> {
        let engine = self.engine.clone();
        self.runtime.block_on(async move {
            let engine = engine.lock().await;
            engine
                .persist()
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn __repr__(&self) -> String {
        let engine = self.engine.clone();
        self.runtime.block_on(async move {
            let engine = engine.lock().await;
            let s = engine.stats().await;
            format!(
                "EchoMemory(memories={}, avg_latency={:.1}ms)",
                s.total_memories, s.avg_echo_latency_ms
            )
        })
    }
}

/// ShrimPK -- The AI memory engine.
#[pymodule]
fn shrimpk(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EchoMemory>()?;
    m.add_class::<PyEchoConfig>()?;
    m.add_class::<PyEchoResult>()?;
    m.add_class::<PyMemoryStats>()?;
    Ok(())
}
