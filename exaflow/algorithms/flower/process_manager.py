import os
import subprocess
import sys
import time

import psutil

try:  # pragma: no cover - fallback for contexts without worker config
    from exaflow.worker import config as worker_config
except Exception:  # noqa: BLE001

    class _FallbackWorkerConfig:  # pragma: no cover - best effort fallback
        class worker_tasks:  # noqa: D401 - simple namespace
            tasks_timeout = 120

    worker_config = _FallbackWorkerConfig()


def _task_timeout_seconds() -> int:
    timeout = getattr(worker_config.worker_tasks, "tasks_timeout", 120)
    try:
        timeout = int(timeout)
    except (TypeError, ValueError):  # noqa: PERF203
        timeout = 120
    return max(1, timeout)


def _deadline(seconds: int | None = None) -> float:
    return time.monotonic() + (
        seconds if seconds is not None else _task_timeout_seconds()
    )


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


def terminate_process(proc, logger, max_attempts=3, deadline: float | None = None):
    """Attempt to terminate the process with retries respecting the worker task timeout."""

    if deadline is None:
        deadline = _deadline()

    for attempt in range(max_attempts):
        status = process_status(proc)
        if status in ["not running", "does not exist"]:
            logger.info(f"Process {proc.pid} is already terminated.")
            return
        if status == "zombie":
            return handle_zombie(proc, logger)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning(f"Stopping PID {proc.pid} exceeded the worker task timeout.")
            return

        signal = "TERM" if attempt < max_attempts - 1 else "KILL"
        # Allow a small wait even if remaining time is tiny to give the OS a chance to reap the process.
        timeout = (
            max(0.5, remaining / (max_attempts - attempt))
            if signal == "TERM"
            else remaining
        )
        if timeout <= 0:
            logger.warning(f"No time left to terminate PID {proc.pid}.")
            return

        if send_signal(proc, signal, timeout, logger):
            return

    logger.error(f"Failed to terminate PID {proc.pid} after {max_attempts} attempts.")


def should_terminate_process(cmdline):
    """Check if the process should be terminated based on its command line."""
    return cmdline and (
        cmdline[-1].endswith("client.py") or cmdline[-1].endswith("server.py")
    )


def process_garbage_collect(proc, logger, deadline):
    """Terminate a process and handle errors."""
    try:
        pid = proc.pid
        cmdline = proc.cmdline()
        if should_terminate_process(cmdline):
            logger.info(f"PID: {pid} - Name: {proc.name()} - Cmdline: {cmdline}")
            terminate_process(proc, logger, deadline=deadline)
    except psutil.NoSuchProcess:
        logger.warn(
            f"No process found with PID {proc.pid}. It may have already exited."
        )
    except psutil.AccessDenied:
        logger.error(f"Access denied when attempting to terminate PID {proc.pid}.")
    except Exception as e:
        logger.error(f"An error occurred while managing PID {proc.pid}: {e}")


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
        env = {**os.environ, **{k: str(v) for k, v in self.env_vars.items()}}
        command = [sys.executable, str(self.file), *self.parameters]
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
                terminate_process(proc, logger, deadline=_deadline())
        except psutil.NoSuchProcess:
            logger.warn(f"No process found with PID {pid}. It may have already exited.")
        except psutil.AccessDenied:
            logger.error(f"Access denied when attempting to terminate PID {pid}.")
        except Exception as e:
            logger.error(f"An error occurred while managing PID {pid}: {e}")

    @classmethod
    def garbage_collect(cls, logger):
        """Garbage collect processes matching specific criteria."""
        deadline = _deadline()
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            if time.monotonic() >= deadline:
                logger.warning(
                    "Flower process garbage collection exceeded worker task timeout."
                )
                break
            process_garbage_collect(proc, logger, deadline=deadline)
