try:
    import msgpack
except: 
    import launch
    launch.run_pip(f"install msgpack", f"msgpack for nai api preview streaming.")