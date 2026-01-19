import torch.nn as nn

def inference_mode(m: nn.Module):
  """Switch model to evaluation mode for inference."""
  return m.eval()

"""
if __name__=="__name__":
    print("file==> ",nn.__file__)
import importlib.metadata as md
for dist in md.distributions():
    try:
        text = dist.read_text("entry_points.txt")
        if text and "\x00" in text:
            print("Corrupted:", dist.metadata["Name"], dist.locate_file("entry_points.txt"))
    except Exception as e:
        print("Error reading", dist, e)
"""