import os
import json
import csv

root = os.getcwd()
outputs_dir = os.path.join(root, "outputs")
rows = []

for name in sorted(os.listdir(outputs_dir)):
    exp_dir = os.path.join(outputs_dir, name)
    if not os.path.isdir(exp_dir):
        continue
    tr = os.path.join(exp_dir, "test_results.json")
    if not os.path.exists(tr):
        continue
    with open(tr, "r", encoding="utf-8") as f:
        data = json.load(f)
    test = data.get("test", {})
    best_epoch = data.get("best_epoch")
    cm = data.get("confusion_matrix", [[None,None],[None,None]])
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    rows.append({
        "experiment": name,
        "best_epoch": best_epoch,
        "test_loss": test.get("loss"),
        "accuracy": test.get("accuracy"),
        "precision": test.get("precision"),
        "recall": test.get("recall"),
        "f1": test.get("f1"),
        "auroc": test.get("auroc"),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    })

csv_path = os.path.join(outputs_dir, "summary_results.csv")
with open(csv_path, "w", newline='', encoding="utf-8") as csvfile:
    fieldnames = ["experiment","best_epoch","test_loss","accuracy","precision","recall","f1","auroc","tn","fp","fn","tp"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"Wrote summary CSV: {csv_path}")

# Try to plot summary (optional)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    experiments = [r['experiment'] for r in rows]
    accuracies = [r['accuracy'] or 0 for r in rows]
    aurocs = [r['auroc'] or 0 for r in rows]

    x = range(len(experiments))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar([i - width/2 for i in x], accuracies, width, label='accuracy')
    ax.bar([i + width/2 for i in x], aurocs, width, label='auroc')
    ax.set_xticks(list(x))
    ax.set_xticklabels(experiments, rotation=30, ha='right')
    ax.set_ylim(0,1)
    ax.set_ylabel('Score')
    ax.set_title('Test Accuracy and AUROC by Experiment')
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(outputs_dir, 'summary_metrics.png')
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")
except Exception as e:
    print("Plotting skipped or failed (matplotlib may be missing):", e)

print("Done")
