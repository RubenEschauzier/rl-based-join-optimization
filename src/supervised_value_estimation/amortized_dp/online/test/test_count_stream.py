import threading
import time

from src.supervised_value_estimation.amortized_dp.online import true_cardinalities as tc


def test_count_stream_counts_tasks_added_while_running(monkeypatch):
    seen = []

    def fake_count(session, endpoint, query, timeout_s):
        time.sleep(0.001)
        seen.append((threading.current_thread().name, endpoint))
        if query == "fail":
            raise ValueError("no count")
        return len(query)

    monkeypatch.setattr(tc, "count_on_endpoint", fake_count)
    stream = tc.CountStream(["http://a", "http://b", "http://a"], 1.0)
    assert len(stream.threads) == 2                      # one thread per distinct endpoint
    stream.add([((0, m), "x" * m) for m in range(1, 50)])
    time.sleep(0.01)
    stream.add([((1, 1), "fail")] + [((1, m), "y" * m) for m in range(2, 30)])
    results = stream.finish()
    assert results == {**{(0, m): m for m in range(1, 50)}, (1, 1): None, **{(1, m): m for m in range(2, 30)}}
    assert {e for _, e in seen} == {"http://a", "http://b"} and stream.seconds > 0
