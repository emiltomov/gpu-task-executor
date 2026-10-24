# Examples

Run each with `gtu -i <definitions> -e <expression>` (omit `-e` to use the `# expr:` line of the definitions file), open them in the UI via **Matrix examples** in the main menu or `F2` in the Matrices screen (the files are embedded in the binary by `src/examples.rs`), or open the definitions with `Ctrl+O`.

| Folder | Definitions | Expressions |
|--------|-------------|-------------|
| `markov/` | `chain.txt` – two-state weather chain | `two_days.txt` (distribution in 2 steps), `steady.txt` (16 steps → steady state `[5/6 1/6]`) |
| `graph/` | `graph.txt` – adjacency matrix of a 5-node graph | `degrees.txt`, `neighbors.txt` (adjacent nodes of node 2), `walks3.txt` (walks of length 3), `reach3.txt` (BFS: nodes reachable from node 0 in 3 hops) |
| `lighting/` | `scene.txt` – facet normals, light, rays; `photo.txt` – real NASA photo (`I = image builtin:earth`) | `shading.txt`, `moved_light.txt`, `raycast.txt`, `photo_enhance.txt`, `photo_blur.txt`, `photo_lift.txt` (before/after shown in the UI) |

```sh
gtu -i examples/graph/graph.txt -e examples/graph/reach3.txt
gtu -i examples/markov/chain.txt -e examples/markov/steady.txt
```
