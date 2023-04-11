import logging
import os
import socket
import time
from subprocess import Popen
from typing import Optional

from distributed import Client


logging.basicConfig(level=logging.INFO)

DASK_PATH = "/opt/conda/bin"
SCHEDULER_IP_TEMPLATE = "tcp://{ip}:8786"
IP_WAIT_TIMEOUT_SECONDS = 200
DASK_WORKER_IP_WAITTIME_SECONDS = 1
DASK_WORKER_SHUTDOWN_WAITTIME_SECONDS = 20
WORKER_PORT = "9000"


def start_daemons(
    master_ip: str,
    scheduler_host: str,
    current_host: str,
    dask_path: Optional[str] = DASK_PATH,
    scheduler_ip_template: Optional[str] = SCHEDULER_IP_TEMPLATE,
) -> None:
    """Configure dask scheduler on the master host and dask workers.

    Args:
          master_ip (str): IPv4 address of the master host.
          scheduler_host (str): master host.
          current_host (str): current host.
          dask_path (str): dask path for scheduler and workers
          scheduler_ip_template (str): dask scheduler address
    """

    cmd_start_scheduler = os.path.join(dask_path, "dask-scheduler")
    cmd_start_worker = os.path.join(dask_path, "dask-worker")
    schedule_conn_string = scheduler_ip_template.format(ip=master_ip)
    if current_host == scheduler_host:
        Popen([cmd_start_scheduler])
        Popen([cmd_start_worker, schedule_conn_string, "--worker-port", WORKER_PORT])  # fix the port
    else:
        Popen([cmd_start_worker, schedule_conn_string, "--worker-port", WORKER_PORT])


def get_ip_from_host(host_name: str) -> str:
    """Get a host name to IPv4 address format string.

    Args:
        host_name (str): host name of an instance from SageMaker DLC.
    Returns:
        IPv4 address format string.
    """

    ip_wait_timeout = IP_WAIT_TIMEOUT_SECONDS
    curr_wait_time_seconds = 0
    ip = None

    while curr_wait_time_seconds < ip_wait_timeout and ip is None:
        try:
            ip = socket.gethostbyname(host_name)
            break
        except socket.gaierror:
            curr_wait_time_seconds += DASK_WORKER_IP_WAITTIME_SECONDS
            time.sleep(DASK_WORKER_IP_WAITTIME_SECONDS)

    if curr_wait_time_seconds >= ip_wait_timeout and ip is None:
        raise RuntimeError(f"Exceeded max wait time of {ip_wait_timeout} seconds for hostname resolution")

    return ip


def retire_workers(client: Client) -> None:
    """Iterate over dask workers in client's scheduler, retire them and wait 20 more seconds.

    Args:
        client (distributed.Client): the entry point for dask distribution.
    """
    worker_status = client.scheduler_info()["workers"]
    worker_list = list()
    for worker, status in worker_status.items():
        worker_list.append(worker)

    client.retire_workers(worker_list, close_workers=True)
    time.sleep(DASK_WORKER_SHUTDOWN_WAITTIME_SECONDS)
