import unittest

from fastapi.testclient import TestClient

from agent.main import app
from agent.python_sandbox import SandboxError, execute_analysis


class PythonSandboxTests(unittest.TestCase):
    def test_runs_bounded_data_analysis(self):
        result = execute_analysis(
            "mean = sum(scores) / len(scores)\nprint(round(mean, 3))",
            {"scores": [0.81, 0.86, 0.9]},
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["output"], "0.857")
        self.assertAlmostEqual(result["variables"]["mean"], 0.8566666666666667)

    def test_rejects_imports_attributes_and_filesystem_access(self):
        for code in (
            "import os",
            "print((1).__class__)",
            "open('x.txt', 'w')",
        ):
            with self.subTest(code=code), self.assertRaises(SandboxError):
                execute_analysis(code)

    def test_bounds_loops_and_output(self):
        with self.assertRaisesRegex(SandboxError, "2000 步"):
            execute_analysis("for i in range(2001):\n    x = i")
        with self.assertRaisesRegex(SandboxError, "执行步数"):
            execute_analysis("for i in range(2000):\n    for j in range(2000):\n        x = i + j")

    def test_rejects_invalid_variables_and_data_size(self):
        with self.assertRaisesRegex(SandboxError, "变量名"):
            execute_analysis("print(x)", {"_private": 1})
        with self.assertRaisesRegex(SandboxError, "50 KB"):
            execute_analysis("print(x)", {"x": "a" * 60_000})

    def test_api_exposes_restricted_runner_and_rejects_unsafe_code(self):
        client = TestClient(app)
        response = client.post(
            "/api/tools/python",
            json={"code": "print(sum(scores))", "variables": {"scores": [1, 2, 3]}},
        )
        denied = client.post("/api/tools/python", json={"code": "import os"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["output"], "6")
        self.assertEqual(denied.status_code, 400)


if __name__ == "__main__":
    unittest.main()
