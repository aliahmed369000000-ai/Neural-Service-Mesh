
import os
class UnsafeNode:
    def process(self, data):
        os.system("ls")
        return {"result": "Executed unsafe command"}
