use crate::matrix::{clean_expression, split_input};

/// A bundled example: definitions and expression files from `examples/`.
pub struct Example {
    pub title: &'static str,
    definitions: &'static str,
    expression: &'static str,
}

impl Example {
    /// Returns `(definitions, expression)` ready for the editors.
    pub fn load(&self) -> (String, String) {
        (
            split_input(self.definitions).0,
            clean_expression(self.expression),
        )
    }
}

macro_rules! example {
    ($title:expr, $dir:literal, $defs:literal, $expr:literal) => {
        Example {
            title: $title,
            definitions: include_str!(concat!("../examples/", $dir, "/", $defs, ".txt")),
            expression: include_str!(concat!("../examples/", $dir, "/", $expr, ".txt")),
        }
    };
}

pub const EXAMPLES: &[Example] = &[
    example!(
        "Markov chain: weather in 2 days",
        "markov",
        "chain",
        "two_days"
    ),
    example!("Markov chain: steady state", "markov", "chain", "steady"),
    example!("Graph: node degrees", "graph", "graph", "degrees"),
    example!(
        "Graph: adjacent nodes of node 2",
        "graph",
        "graph",
        "neighbors"
    ),
    example!("Graph: walks of length 3", "graph", "graph", "walks3"),
    example!(
        "Graph: BFS reach from node 0 in 3 hops",
        "graph",
        "graph",
        "reach3"
    ),
    example!("Lighting: Lambert shading", "lighting", "scene", "shading"),
    example!("Lighting: moved light", "lighting", "scene", "moved_light"),
    example!(
        "Lighting: raycast facets vs. rays",
        "lighting",
        "scene",
        "raycast"
    ),
    example!(
        "Photo: enhance contrast/brightness",
        "lighting",
        "photo",
        "photo_enhance"
    ),
    example!("Photo: blur", "lighting", "photo", "photo_blur"),
    example!("Photo: lift shadows", "lighting", "photo", "photo_lift"),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::Gpu;
    use crate::matrix::evaluate;

    #[test]
    fn every_example_evaluates_on_gpu() {
        let Ok(gpu) = Gpu::new() else { return };
        for ex in EXAMPLES {
            let (defs, expr) = ex.load();
            evaluate(&defs, &expr, &gpu).unwrap_or_else(|e| panic!("{}: {e:#}", ex.title));
        }
    }
}
