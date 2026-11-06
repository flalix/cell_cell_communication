"""Execute a notebook in-process with IPython (used where ZMQ kernels cannot start, e.g. a network sandbox).
Usage: python _run_nb_inprocess.py in.ipynb out.ipynb [--stop-on-error]"""
import sys, time, traceback, os
import nbformat as nbf
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

src, dst = sys.argv[1], sys.argv[2]
stop = "--stop-on-error" in sys.argv
os.chdir(os.path.dirname(os.path.abspath(src)))
nb = nbf.read(src, as_version=4)
class _Shell(InteractiveShell):
    def enable_gui(self, gui=None):
        pass

shell = _Shell.instance()
shell.enable_matplotlib("inline")
shell.displayhook.write_output_prompt = lambda: None          # results are stored as execute_result, not echoed
shell.displayhook.write_format_data = lambda *a, **k: None
n = 0
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    n += 1
    t0 = time.time()
    outputs = []
    with capture_output(display=True) as cap:
        res = shell.run_cell(cell.source, store_history=True)
    if cap.stdout:
        outputs.append(nbf.v4.new_output("stream", name="stdout", text=cap.stdout))
    if cap.stderr:
        err = "\n".join(l for l in cap.stderr.splitlines() if "Warning" not in l and "warn(" not in l)
        if err.strip():
            outputs.append(nbf.v4.new_output("stream", name="stderr", text=err))
    for o in cap.outputs:
        outputs.append(nbf.v4.new_output("display_data", data=o.data, metadata=o.metadata or {}))
    if res.result is not None:
        data, md = shell.display_formatter.format(res.result)
        outputs.append(nbf.v4.new_output("execute_result", data=data, metadata=md, execution_count=n))
    ok = res.success
    if not ok:
        e = res.error_in_exec or res.error_before_exec
        outputs.append(nbf.v4.new_output("error", ename=type(e).__name__, evalue=str(e),
                                         traceback=traceback.format_exception(type(e), e, e.__traceback__)))
    cell.outputs = outputs
    cell.execution_count = n
    print(f"cell {n:2d} {'ok ' if ok else 'ERR'} {time.time() - t0:6.1f}s  {cell.source.splitlines()[0][:70] if cell.source else ''}", flush=True)
    if not ok:
        print("   ", type(e).__name__, str(e)[:300], flush=True)
        if stop:
            break
nbf.write(nb, dst)
print("written", dst)
