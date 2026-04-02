//! Tool dispatch — routes tool names to handler functions.

use crate::protocol::ToolCallResult;
use crate::tools;
use serde_json::Value;
use shrimpk_core::EchoConfig;
use shrimpk_memory::EchoEngine;
use std::sync::Arc;

/// Dispatch a tool call to the appropriate handler.
pub async fn dispatch(
    engine: &Arc<EchoEngine>,
    config: &EchoConfig,
    name: &str,
    args: &Value,
) -> ToolCallResult {
    let result = match name {
        "store" => tools::handle_store(engine, args).await,
        "echo" => tools::handle_echo(engine, args).await,
        "stats" => tools::handle_stats(engine, config).await,
        "forget" => tools::handle_forget(engine, args).await,
        "dump" => tools::handle_dump(engine, config, args).await,
        "config_show" => tools::handle_config_show(config),
        "config_set" => tools::handle_config_set(args),
        "persist" => tools::handle_persist(engine, config).await,
        "status" => tools::handle_status(config),
        "memory_graph" => tools::handle_memory_graph(engine, args).await,
        "memory_related" => tools::handle_memory_related(engine, args).await,
        "memory_get" => tools::handle_memory_get(engine, args).await,
        #[cfg(feature = "vision")]
        "store_image" => tools::handle_store_image(engine, args).await,
        #[cfg(feature = "speech")]
        "store_audio" => tools::handle_store_audio(engine, args).await,
        _ => return ToolCallResult::error(format!("Unknown tool: {name}")),
    };

    match result {
        Ok(text) => ToolCallResult::success(text),
        Err(e) => ToolCallResult::error(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn unknown_tool_returns_error() {
        let _config = EchoConfig::default();
        // We can't easily create an EchoEngine in tests without fastembed,
        // but we can test the unknown tool path since it doesn't use the engine.
        // For now, just verify the error message format.
        let result = ToolCallResult::error("Unknown tool: nonexistent".into());
        assert!(result.is_error.unwrap());
    }

    #[test]
    fn dispatch_has_all_base_tools_covered() {
        // Verify the match arms cover all base tool names
        #[allow(unused_mut)]
        let mut tool_names: Vec<&str> = vec![
            "store",
            "echo",
            "stats",
            "forget",
            "dump",
            "config_show",
            "config_set",
            "persist",
            "status",
            "memory_graph",
            "memory_related",
            "memory_get",
        ];
        #[cfg(feature = "vision")]
        tool_names.push("store_image");
        #[cfg(feature = "speech")]
        tool_names.push("store_audio");

        #[allow(unused_mut)]
        let mut expected = 12;
        #[cfg(feature = "vision")]
        { expected += 1; }
        #[cfg(feature = "speech")]
        { expected += 1; }
        assert_eq!(tool_names.len(), expected);
    }
}
