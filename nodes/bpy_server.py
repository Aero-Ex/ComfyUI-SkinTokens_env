import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.server.bpy_server import run

def main():
    run()

if __name__ == "__main__":
    main()
