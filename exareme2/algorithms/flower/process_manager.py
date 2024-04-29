import os
import subprocess
from pathlib import Path

import psutil

ALGORITHMS_ROOT = Path(__file__).parent


def process_status(proc):
    """Check the status of a process."""
    try:
        if not proc.is_running():
            return "not running"
        if proc.status() == psutil.STATUS_ZOMBIE:
            return "zombie"
        return "running"
    except psutil.NoSuchProcess:
        return "does not exist"


def handle_zombie(proc, logger):
    """Clean up a zombie process."""
    if proc.status() == psutil.STATUS_ZOMBIE:
        logger.info(f"Attempting to reap zombie process {proc.pid}")
        try:
            os.waitpid(proc.pid, 0)
            logger.info(f"Zombie process {proc.pid} reaped successfully.")
        except ChildProcessError:
            logger.error(
                f"Zombie process {proc.pid} could not be reaped. It may have been reaped already."
            )
        except Exception as e:
            logger.error(f"Error reaping zombie PID {proc.pid}: {e}")


def send_signal(proc, signal, timeout, logger):
    """Send a signal to the process and wait for it to terminate."""
    try:
        (proc.terminate if signal == "TERM" else proc.kill)()
        proc.wait(timeout=timeout)
        logger.info(f"Process {proc.pid} terminated with {signal}.")
        return True
    except psutil.TimeoutExpired:
        logger.warning(
            f"Process {proc.pid} did not terminate after {signal}. Trying again..."
        )
        return False
    except psutil.NoSuchProcess:
        logger.info(f"Process {proc.pid} no longer exists.")
        return True


def terminate_process(proc, logger, max_attempts=3, wait_time=10):
    """Attempt to terminate the process with a limited number of retries, handling zombie processes."""

    for attempt in range(max_attempts):
        status = process_status(proc)
        if status in ["not running", "does not exist"]:
            logger.info(f"Process {proc.pid} is already terminated.")
            return
        if status == "zombie":
            return handle_zombie(proc, logger)

        signal = "TERM" if attempt < max_attempts - 1 else "KILL"
        timeout = wait_time if signal == "TERM" else None
        if send_signal(proc, signal, timeout, logger):
            return

    logger.error(f"Failed to terminate PID {proc.pid} after {max_attempts} attempts.")


class FlowerProcess:
    def __init__(self, file, parameters=None, env_vars=None, stdout=None, stderr=None):
        self.file = file
        self.parameters = parameters if parameters is not None else []
        self.env_vars = env_vars if env_vars is not None else {}
        self.stdout = stdout
        self.stderr = stderr
        self.proc = None

    def start(self, logger):
        if self.proc is not None:
            logger.error("Process already started!")
            raise RuntimeError("Process already started!")
        flower_executable = ALGORITHMS_ROOT / self.file
        env = {**os.environ, **{k: str(v) for k, v in self.env_vars.items()}}
        command = ["poetry", "run", "python", str(flower_executable), *self.parameters]
        logger.info(f"Executing command: {command}")
        self.proc = subprocess.Popen(
            command, env=env, stdout=self.stdout, stderr=self.stderr
        )
        return self.proc.pid

    @classmethod
    def kill_process(cls, pid, algorithm_name, logger):
        """Terminate a process based on the algorithm name, with logging."""
        try:
            proc = psutil.Process(pid)
            command_line = " ".join(proc.cmdline())
            logger.info(f"Command line for PID {pid}: {command_line}")
            if algorithm_name.lower() in command_line.lower():
                terminate_process(proc, logger)
        except psutil.NoSuchProcess:
            logger.warn(f"No process found with PID {pid}. It may have already exited.")
        except psutil.AccessDenied:
            logger.error(f"Access denied when attempting to terminate PID {pid}.")
        except Exception as e:
            logger.error(f"An error occurred while managing PID {pid}: {e}")
