import numpy as np

from tools.openai_study_drift import learn_shift, summarize_windows


class FixedToy:
    arms = ('cheap', 'strong')
    A = np.asarray([np.eye(1), np.eye(1)])
    rounds = 0

    def select(self, x):
        return 'cheap'

    def scores(self, x):
        return dict(cheap=1., strong=0.)

    def observe(self, arm, x, reward):
        assert arm == 'cheap'
        self.rounds += 1


def test_shift_observes_only_selected_arm_and_counts_warm_rounds_separately():
    model = FixedToy()
    calls = []
    def selected(q, arm):
        calls.append((q, arm))
        assert arm == 'cheap'
        return (float(q != 'a'), .1)
    rows = learn_shift(model, {'warm':['w'], 'before':['b'], 'after':['a']},
        lambda q:np.asarray([1.]), selected, lambda q:1.)
    assert calls == [('w','cheap'), ('b','cheap'), ('a','cheap')]
    assert [r['rounds_before_feedback'] for r in rows] == [0, 1, 2]
    assert [r['phase'] for r in rows] == ['warm', 'before', 'after']
    assert rows[-1]['reward'] == 0 and rows[-1]['empirical_oracle_regret'] == 1


def test_drift_window_denominators_exclude_warm_and_keep_failed_answers():
    rows = [dict(phase='warm', reward=1., actual_cost=9., gpt4_reward=1., gpt4_cost=9.),
            dict(phase='before', reward=1., actual_cost=.1, gpt4_reward=1., gpt4_cost=1.),
            dict(phase='before', reward=0., actual_cost=.1, gpt4_reward=1., gpt4_cost=1.),
            dict(phase='after', reward=0., actual_cost=.1, gpt4_reward=1., gpt4_cost=1.)]
    windows = summarize_windows(rows, size=1)
    assert windows['all-evaluation']['requested'] == 3
    assert windows['all-evaluation']['quality_fraction_gpt4'] == 1/3
    assert windows['last-before']['quality_fraction_gpt4'] == 0
    assert windows['first-after']['coverage'] == 1
