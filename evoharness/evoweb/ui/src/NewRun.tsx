import { useState } from "react";
import { createRun, RECIPES, TASKS } from "./api";

const field = {
  width: "100%",
  font: "inherit",
  padding: "5px 8px",
  border: "1px solid var(--rule)",
  borderRadius: 6,
  marginBottom: 8,
} as const;

export function NewRun({ onStarted }: { onStarted: () => void }) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [recipe, setRecipe] = useState("e3r");
  const [task, setTask] = useState("demo_counter");
  const [generations, setGenerations] = useState(10);
  const [model, setModel] = useState("qwen3-coder-plus");
  const [live, setLive] = useState(false);
  const [error, setError] = useState("");

  if (!open)
    return (
      <button className="btn-primary" style={{ width: "100%", marginBottom: 16 }}
              onClick={() => setOpen(true)}>
        New run
      </button>
    );

  const start = async () => {
    setError("");
    try {
      await createRun({
        name, recipe, task, generations,
        model: live ? model : undefined, live,
      });
      setOpen(false);
      setName("");
      onStarted();
    } catch (e) {
      setError(String(e));
    }
  };

  return (
    <div style={{ background: "var(--card)", border: "1px solid var(--rule)",
                  borderRadius: 8, padding: 12, marginBottom: 16,
                  fontSize: 12.5 }}>
      <input style={field} placeholder="run name (e.g. e3r_seed2)"
             value={name} onChange={(e) => setName(e.target.value)} />
      <select style={field} value={recipe}
              onChange={(e) => setRecipe(e.target.value)}>
        {RECIPES.map((r) => <option key={r}>{r}</option>)}
      </select>
      <select style={field} value={task}
              onChange={(e) => setTask(e.target.value)}>
        {TASKS.map((t) => <option key={t}>{t}</option>)}
      </select>
      <input style={field} type="number" min={1} max={500} value={generations}
             onChange={(e) => setGenerations(Number(e.target.value))} />
      <label style={{ display: "block", marginBottom: 8 }}>
        <input type="checkbox" checked={live}
               onChange={(e) => setLive(e.target.checked)} />{" "}
        live LLM (needs EVOHARNESS_API_* env)
      </label>
      {live && (
        <input style={field} placeholder="model" value={model}
               onChange={(e) => setModel(e.target.value)} />
      )}
      {error && <p className="tag-dup">{error}</p>}
      <div style={{ display: "flex", gap: 8 }}>
        <button className="btn-primary" onClick={start}>Start run</button>
        <button onClick={() => setOpen(false)}>Cancel</button>
      </div>
    </div>
  );
}
