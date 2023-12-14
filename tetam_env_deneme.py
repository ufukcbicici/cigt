import torch
import numpy as np

from auxillary.db_logger import DbLogger

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("Device:{0}".format(device))
x = torch.rand(size=(10, 5)).to(device)
print(x.device)
y = np.random.uniform(size=(100, 500))
print(np.mean(y))

DbLogger.log_db_path = DbLogger.hpc_docker1
run_id = DbLogger.get_run_id()
print("RunId is:{0}".format(run_id))
DbLogger.write_into_table(
    rows=[(run_id,
           42,
           67,
           94.32,
           0.054,
           94.38,
           0.062,
           0.1245)], table=DbLogger.logsTableQCigt)
