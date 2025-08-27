python - <<'PY'
import inspect, pathlib, time, sys
import lm_eval

root = pathlib.Path(inspect.getfile(lm_eval)).parent
target = root / "models" / "vllm_causallms.py"
print(f"[info] target: {target}")

text = target.read_text(encoding="utf-8")
if 'ModelRegistry.register_model("TTLlamaForCausalLM"' in text:
    print("[skip] Already patched.")
    sys.exit(0)

bak = f"{target}.bak"
pathlib.Path(bak).write_text(text, encoding="utf-8")
print(f"[ok] backup -> {bak}")

insert_block = (
    "from vllm import ModelRegistry\n"
    "path_llama_text='models.tt_transformers.tt.generator_vllm:LlamaForCausalLM'\n"
    "ModelRegistry.register_model(\"TTLlamaForCausalLM\", path_llama_text)\n"
)
lines = text.splitlines(keepends=True)
idx = 30 if len(lines) >= 30 else len(lines)
new_text = "".join(lines[:idx] + [insert_block] + lines[idx:])

target.write_text(new_text, encoding="utf-8")
print("[ok] patched at line", idx + 1)

new_text = target.read_text(encoding="utf-8")
assert 'ModelRegistry.register_model("TTLlamaForCausalLM"' in new_text
print("[ok] verify: TTLlamaForCausalLM registered line inserted.")
PY