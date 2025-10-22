from pathlib import Path
import matplotlib.pyplot as plt

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_line(y, title, out_path):
    ensure_dir(Path(out_path).parent)
    plt.figure()
    plt.plot(y)
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
