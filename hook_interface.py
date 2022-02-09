import subprocess
import sys
from importlib import import_module


class HookInterface:
    def __init__(self):

        self.modifieds = []
        self.get_modified_files()

    def get_modified_files(self):
        """
        finds the files that were modified (excludes added or deleted files)
        """
        command = subprocess.Popen(['git', 'status', '--porcelain'],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        output, error = command.communicate()
        if error.decode('utf-8'):
            print('error occurred by --- git status ---')
            sys.exit(-1)
        output = output.decode('utf-8').splitlines()

        self.modifieds = []
        for line in output:
            change = line.split()
            change_type = change[0]
            if change_type == 'M':
                path = change[1]
                self.modifieds.append(path)

    def run_model(self):
        raise NotImplementedError("please implement this method")


if __name__ == '__main__':
    module = 'jitgnn.jitgnn_hook'
    model = 'JITGNNHook'
    mod = import_module(module)
    cls = getattr(mod, model)
    hook = cls()
    hook.run_model()


