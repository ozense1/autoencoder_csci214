"""Streamlit integration with the Matryoshka baseline present."""
from streamlit.testing.v1 import AppTest
from test_app import APP, ui_environment, ready_environment
from storage import load_registry
from matryoshka import build_matryoshka

def test_ui_with_matryoshka(ui_environment):
    ready_environment(ui_environment)
    artifacts = ui_environment[1]
    build_matryoshka('matryoshka_128', 4, load_registry(artifacts)['baseline'], artifacts, progress=lambda _: None)
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    at.radio(key='view').set_value('Compare methods').run()
    at.multiselect(key='compare_methods').set_value(['baseline', 'matryoshka_128', 'mse']).run()
    at.button(key='search').click().run()
    assert not at.exception and not at.error
    at.button(key='evaluate').click().run()
    assert not at.exception and not at.error
    assert set(at.session_state['evaluation_report']['methods']) == {'baseline', 'matryoshka_128', 'mse'}
    assert at.selectbox(key="training_method").options == ["Embedding + AE: MSE"]  # Matryoshka has no training run
