import runpy
import sys

# Передаём аргументы как будто мы запустили их из командной строки
sys.argv = [
    "scripts/run_mot_challenge.py",
    "--BENCHMARK", "mipt",
    "--TRACKERS_TO_EVAL", "CarSORT-botkalman-stopping",
    "--METRICS", "HOTA", "CLEAR", "Identity",
    "--USE_PARALLEL", "True",
    "--NUM_PARALLEL_CORES", "8",
    "--DO_PREPROC", "False"
]

# Запускаем основной скрипт TrackEval
runpy.run_path("scripts/run_mot_challenge.py", run_name="__main__")
