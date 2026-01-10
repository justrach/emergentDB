//! Tool definitions for LLM agents.

use serde::{Deserialize, Serialize};

/// Tool definition for LLM function calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: ToolParameters,
}

/// Parameters schema for a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParameters {
    #[serde(rename = "type")]
    pub param_type: String,
    pub properties: serde_json::Value,
    #[serde(default)]
    pub required: Vec<String>,
}

impl ToolDefinition {
    /// Create the vector_search tool definition.
    pub fn vector_search() -> Self {
        Self {
            name: "vector_search".to_string(),
            description: "Search for semantically similar vectors in the index. Returns the k nearest neighbors.".to_string(),
            parameters: ToolParameters {
                param_type: "object".to_string(),
                properties: serde_json::json!({
                    "query": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "The query vector (embedding)"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 10
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["cosine", "euclidean", "dot_product"],
                        "description": "Distance metric to use",
                        "default": "cosine"
                    }
                }),
                required: vec!["query".to_string()],
            },
        }
    }

    /// Create the graph_query tool definition.
    pub fn graph_query() -> Self {
        Self {
            name: "graph_query".to_string(),
            description: "Query the context graph to find related concepts through semantic relationships.".to_string(),
            parameters: ToolParameters {
                param_type: "object".to_string(),
                properties: serde_json::json!({
                    "start_id": {
                        "type": "string",
                        "description": "UUID of the starting concept"
                    },
                    "relation_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Types of relationships to follow (IS_A, PART_OF, RELATED_TO, etc.)"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum traversal depth",
                        "default": 3
                    }
                }),
                required: vec!["start_id".to_string()],
            },
        }
    }

    /// Create the research tool definition.
    pub fn research() -> Self {
        Self {
            name: "research".to_string(),
            description: "Perform deep research on a topic. Returns a synthesized answer with sources.".to_string(),
            parameters: ToolParameters {
                param_type: "object".to_string(),
                properties: serde_json::json!({
                    "query": {
                        "type": "string",
                        "description": "The research query"
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["basic", "medium", "default", "deeper"],
                        "description": "Research depth tier (affects cost and latency)",
                        "default": "default"
                    },
                    "max_sources": {
                        "type": "integer",
                        "description": "Maximum number of sources to return",
                        "default": 10
                    }
                }),
                required: vec!["query".to_string()],
            },
        }
    }

    /// Create the qd_diversify tool definition.
    pub fn qd_diversify() -> Self {
        Self {
            name: "qd_diversify".to_string(),
            description: "Generate diverse query variations using MAP-Elites quality-diversity algorithm.".to_string(),
            parameters: ToolParameters {
                param_type: "object".to_string(),
                properties: serde_json::json!({
                    "base_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Seed queries to diversify"
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of diverse variations to generate",
                        "default": 10
                    },
                    "behavior_dimensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Dimensions for behavior characterization",
                        "default": ["specificity", "domain"]
                    }
                }),
                required: vec!["base_queries".to_string()],
            },
        }
    }

    /// Get all available tool definitions.
    pub fn all_tools() -> Vec<Self> {
        vec![
            Self::vector_search(),
            Self::graph_query(),
            Self::research(),
            Self::qd_diversify(),
        ]
    }
}

/// Convert tool definitions to OpenAI function calling format.
pub fn to_openai_functions(tools: &[ToolDefinition]) -> Vec<serde_json::Value> {
    tools
        .iter()
        .map(|tool| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": tool.parameters.param_type,
                        "properties": tool.parameters.properties,
                        "required": tool.parameters.required
                    }
                }
            })
        })
        .collect()
}

/// Convert tool definitions to Anthropic tool use format.
pub fn to_anthropic_tools(tools: &[ToolDefinition]) -> Vec<serde_json::Value> {
    tools
        .iter()
        .map(|tool| {
            serde_json::json!({
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": tool.parameters.param_type,
                    "properties": tool.parameters.properties,
                    "required": tool.parameters.required
                }
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_definitions() {
        let tools = ToolDefinition::all_tools();
        assert_eq!(tools.len(), 4);

        let names: Vec<_> = tools.iter().map(|t| &t.name).collect();
        assert!(names.contains(&&"vector_search".to_string()));
        assert!(names.contains(&&"research".to_string()));
    }

    #[test]
    fn test_openai_format() {
        let tools = ToolDefinition::all_tools();
        let openai = to_openai_functions(&tools);

        assert_eq!(openai.len(), 4);
        assert_eq!(openai[0]["type"], "function");
    }

    #[test]
    fn test_anthropic_format() {
        let tools = ToolDefinition::all_tools();
        let anthropic = to_anthropic_tools(&tools);

        assert_eq!(anthropic.len(), 4);
        assert!(anthropic[0]["input_schema"].is_object());
    }
}
