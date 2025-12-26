import sys

sys.path.append("../libact-dev/")
try:
    from libact_dev.query_strategies import QueryByCommittee as libact_QBC

    print("Successfully imported libact_QBC")
except Exception as e:
    print(f"Failed to import libact_QBC: {e}")
