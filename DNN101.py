from pathlib import Path
import torch.nn.functional as f
import numpy 
import torch
import textwrap
import bootstrap
import subprocess
import math

# def clone_repo(repo,branch,prefix=""):
#     url = f"https://{prefix}github.com/full-stack-deep-learning/{repo}"
#     subprocess.run(["git","clone","--branch",branch,"-q",url],check=True)
#     
# if __name__=="__main__":
#     repo = "fsdl-text-recognizer-2022-labs"
#     branch = "main"
#     clone_repo(repo,branch)

weights = torch.randn(784,10)/math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10,requires_grad=True)
print(weights.shape)
print(bias.shape)
