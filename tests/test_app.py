"""Exercise the actual local Streamlit UI, without a browser or fake backend."""
from pathlib import Path
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'app.py'


def test_default_form_runs_and_retains_finite_session():
    assert APP.is_file(), 'Streamlit application is not implemented yet'
    at = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not at.exception
    assert at.text_input(key='amplitudes').value == '1, 0.5, 0.25'
    assert at.text_input(key='frequencies').value == '0.05, 0.1, 0.4'
    assert not at.get('plotly_chart')
    assert at.number_input(key='ema_period').value == 3
    assert any('Период 3 выбран экспериментально на контрольном сеансе; настраивается' in item.value
               for item in at.caption)
    at.button[0].click().run()
    assert not at.exception
    assert not at.error
    result = at.session_state['result']
    assert len(result.t) == 1200
    np.testing.assert_array_equal(result.t, np.arange(1200) / 20)
    assert len(at.get('plotly_chart')) == 2
    assert len(at.dataframe[0].value) == 1200
    assert any('НЕ обнаружение' in item.value for item in at.caption)
    previous = result.raw.copy()
    at.run()
    np.testing.assert_array_equal(at.session_state['result'].raw, previous)


@pytest.mark.parametrize('widget, key, invalid', [
    ('text_input', 'amplitudes', 'not-a-number'),
    ('text_input', 'amplitudes', '1, 2'),
    ('text_input', 'frequencies', 'nan, 0.1, 0.4'),
    ('number_input', 'fs', 0.0),
    ('number_input', 'fs', 0.8),
    ('number_input', 'duration', 1.001),
    ('number_input', 'ema_period', 0),
    ('number_input', 'seed', -1),
])
def test_invalid_submit_clears_old_result_and_can_recover(widget, key, invalid):
    at = AppTest.from_file(str(APP), default_timeout=20).run()
    at.button[0].click().run()
    assert at.session_state['result'] is not None
    control = getattr(at, widget)(key=key)
    original = control.value
    control.set_value(invalid)
    at.button[0].click().run()
    assert not at.exception
    assert at.error
    assert 'Некорректные параметры' in at.error[0].value
    assert at.session_state['result'] is None
    assert not at.get('plotly_chart')
    assert not at.dataframe
    getattr(at, widget)(key=key).set_value(original)
    at.button[0].click().run()
    assert not at.error
    assert not at.exception
    assert at.session_state['result'] is not None


def test_saturation_is_visible_to_operator():
    at = AppTest.from_file(str(APP), default_timeout=20).run()
    at.text_input(key='amplitudes').set_value('10, 5, 2.5')
    at.button[0].click().run()
    assert not at.exception
    count = at.session_state['result'].saturation_count
    assert count > 0
    assert at.warning
    assert str(count) in at.warning[0].value
    assert 'Насыщение АЦП' in at.warning[0].value
