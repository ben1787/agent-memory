"""PyInstaller entrypoint shim.

PyInstaller wants a concrete .py file as the build entrypoint. This file
exists solely to call into agent_memory.cli.main(); the binary's behavior is
defined entirely by the agent_memory package.
"""

from agent_memory.cli import main


if __name__ == "__main__":
    main()
