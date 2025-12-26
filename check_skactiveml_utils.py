try:
    import skactiveml.pool.utils as utils

    print("Contents of skactiveml.pool.utils:")
    print(dir(utils))
except ImportError as e:
    print(f"Error importing skactiveml.pool.utils: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
