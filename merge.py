import os
import re
import numpy as np
import time


base_dir = '/share/project/lijijie/dzj/DR/DRAGON_V2/test_result'

sub_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
latest_dir = max(sub_dirs, key=lambda d: time.strptime(d, "%Y%m%d-%H%M%S"))

log_dir = os.path.join(base_dir, latest_dir)

log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith('.txt')] 

print(len(log_files))

auroc_values = []
aupr_values = []

for log_file in log_files:
    with open(log_file, 'r') as f:
        lines = f.readlines()[-2:]

        auroc_match = re.search(r'auroc：(\d+\.\d+)', lines[0])
        aupr_match = re.search(r'aupr：(\d+\.\d+)', lines[1])


        if auroc_match and aupr_match:
            auroc = float(auroc_match.group(1))
            aupr = float(aupr_match.group(1))

            auroc_values.append(auroc)
            aupr_values.append(aupr)

avg_auroc = np.mean(auroc_values)
std_auroc = np.std(auroc_values)
avg_aupr = np.mean(aupr_values)
std_aupr = np.std(aupr_values)

print(f"AUROC mean：{avg_auroc:.5f}, std：{std_auroc:.5f}")
print(f"AUPR mean：{avg_aupr:.5f}, std：{std_aupr:.5f}")

result_file = os.path.join(log_dir, 'result.txt')
with open(result_file, 'w') as f:
    f.write(f"Average AUROC: {avg_auroc:.5f}, Std: {std_auroc:.5f}\n")
    f.write(f"Average AUPR: {avg_aupr:.5f}, Std: {std_aupr:.5f}\n")
    f.write("Individual trial results: \n")
    for i in range(len(auroc_values)):
        f.write(f"{i + 1}-th trial AUROC: {auroc_values[i]:.5f}, AUPR: {aupr_values[i]:.5f}\n")