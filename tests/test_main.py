import unittest


def hello_world() -> str:
    """Returns a greeting message."""
    return "Hello, World!"


class MainTest(unittest.TestCase):
    def test_hello(self) -> None:
        self.assertEqual(hello_world(), "Hello, World!")


if __name__ == "__main__":
    unittest.main()
