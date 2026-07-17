"""Worker entrypoint: a long-lived process that pulls jobs off the queue and runs them.

    python -m app.worker

Run several of these (one per process) to get bounded concurrency -- each one loads its own
copy of the models (~2-3GB RAM; see the models loaded by app.analysis.result.build_result),
so the number of workers a host can run is capped by RAM, not just CPU. Models load lazily on
first use and stay cached for the life of the process, so only the first job per worker pays
the load cost.
"""

from __future__ import annotations

from rq import Worker

from app.config import QUEUE_NAME
from app.queue import get_redis


def main() -> None:
    worker = Worker([QUEUE_NAME], connection=get_redis())
    # with_scheduler=True: retries (job.retry with intervals) are implemented via RQ's
    # scheduler moving jobs from the scheduled registry back onto the queue at the right
    # time -- without this, a retrying job sits in "scheduled" forever.
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
