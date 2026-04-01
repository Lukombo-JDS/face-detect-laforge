from __future__ import annotations

import time
import unittest

from app.workers.background import BackgroundTaskRunner


class WorkerTests(unittest.TestCase):
    def test_task_runner_executes_job(self):
        runner = BackgroundTaskRunner()
        task_id = runner.submit(lambda x: x + 1, 1)
        for _ in range(20):
            status = runner.status(task_id)
            if status.status == "done":
                break
            time.sleep(0.05)

        self.assertEqual(runner.status(task_id).result, 2)
        runner.shutdown()


if __name__ == "__main__":
    unittest.main()
