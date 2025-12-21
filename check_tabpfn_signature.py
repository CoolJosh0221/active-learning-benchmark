
import inspect
from tabpfn import TabPFNClassifier

print("TabPFNClassifier __init__ signature:")
try:
    print(inspect.signature(TabPFNClassifier.__init__))
except Exception as e:
    print(e)

print("\nTabPFNClassifier help:")
help(TabPFNClassifier)
