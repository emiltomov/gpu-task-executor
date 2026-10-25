# Examples

Run each with `gtu -i <definitions> -e <expression>` (omit `-e` to use the `# expr:` line of the definitions file), open them in the UI via **Matrix examples** in the main menu or `F2` in the Matrices screen (the files are embedded in the binary by `src/examples.rs`), or open the definitions with `Ctrl+O`.

| Folder | Definitions | Expressions |
|--------|-------------|-------------|
| `markov/` | `chain.txt` – two-state weather chain | `two_days.txt` (distribution in 2 steps), `steady.txt` (16 steps → steady state `[5/6 1/6]`) |
| `graph/` | `graph.txt` – adjacency matrix of a 5-node graph | `degrees.txt`, `neighbors.txt` (adjacent nodes of node 2), `walks3.txt` (walks of length 3), `reach3.txt` (BFS: nodes reachable from node 0 in 3 hops) |
| `maze/` | `maze.txt` – random maze of `W = 21` x `H = 24` cells (504 nodes; edit the variables to resize; `M = maze W H [seed]`), drawn on a grid (`graph M grid W`) | `astar.txt` (`astar(M, 0, W * H - 1, W)`: shortest path, highlighted yellow; the graph also shows the longest path in green and dead ends in red), `discover.txt` (`bfs(M, 0)`: distance to every cell) |
| `lighting/` | `scene.txt` – facet normals, light, rays; `photo.txt` – real NASA photo (`I = image builtin:earth`) | `shading.txt`, `moved_light.txt`, `photo_raycast.txt` (camera rotated 40°: one ray per pixel resamples the photo, black where rays miss), `photo_enhance.txt`, `photo_blur.txt`, `photo_lift.txt`, `photo_sharpen.txt`, `photo_edges.txt`, `photo_invert.txt`, `photo_scale_down.txt`, `photo_scale_up.txt` (`resize`), `photo_flip.txt`, `photo_rotate.txt` (`flip(n)` exchange matrix) (before/after shown in the UI; most have a `# back:` inverse for Ctrl+P) |

```sh
gtu -i examples/graph/graph.txt -e examples/graph/reach3.txt
gtu -i examples/markov/chain.txt -e examples/markov/steady.txt
```
