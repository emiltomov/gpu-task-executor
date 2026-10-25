use crate::matrix::{back_expression, clean_expression, split_input};

/// A bundled example: definitions and expression files from `examples/`.
pub struct Example {
    pub title: &'static str,
    definitions: &'static str,
    expression: &'static str,
}

impl Example {
    /// The expression that undoes the example's one (for image examples), or empty.
    pub fn back(&self) -> String {
        back_expression(self.expression).unwrap_or_default()
    }

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
    example!("Maze: A* path through 504 cells", "maze", "maze", "astar"),
    example!(
        "Maze: discover every cell (BFS)",
        "maze",
        "maze",
        "discover"
    ),
    example!("Lighting: Lambert shading", "lighting", "scene", "shading"),
    example!("Lighting: moved light", "lighting", "scene", "moved_light"),
    example!(
        "Photo: raycast (camera rotated 40 degrees)",
        "lighting",
        "photo",
        "photo_raycast"
    ),
    example!(
        "Photo: enhance contrast/brightness",
        "lighting",
        "photo",
        "photo_enhance"
    ),
    example!("Photo: blur", "lighting", "photo", "photo_blur"),
    example!("Photo: lift shadows", "lighting", "photo", "photo_lift"),
    example!("Photo: sharpen", "lighting", "photo", "photo_sharpen"),
    example!("Photo: edges", "lighting", "photo", "photo_edges"),
    example!("Photo: negative", "lighting", "photo", "photo_invert"),
    example!(
        "Photo: scale down (half size)",
        "lighting",
        "photo",
        "photo_scale_down"
    ),
    example!(
        "Photo: scale up (double size)",
        "lighting",
        "photo",
        "photo_scale_up"
    ),
    example!(
        "Photo: mirror left/right",
        "lighting",
        "photo",
        "photo_flip"
    ),
    example!(
        "Photo: rotate 90 degrees",
        "lighting",
        "photo",
        "photo_rotate"
    ),
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
