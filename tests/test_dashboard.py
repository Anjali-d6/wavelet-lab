import numpy as np
from streamlit.testing.v1 import AppTest


def app_with_image(page: str) -> AppTest:
    app = AppTest.from_file("app.py")
    image = np.tile(np.linspace(108, 148, 64, dtype=np.uint8), (64, 1))
    app.session_state["decoded_image"] = image
    app.session_state["upload_filename"] = "synthetic.png"
    app.session_state["upload_fingerprint"] = "synthetic-image"
    app.session_state["upload_size"] = 256
    app.run(timeout=45)
    app.radio[0].set_value(page).run(timeout=45)
    assert not app.exception
    return app


def test_enhancement_workspace_runs_and_shows_before_after_measurements():
    app = app_with_image("Enhance Image")
    run_button = next(button for button in app.button if button.label == "Run enhancement")
    run_button.click().run(timeout=45)

    result = app.session_state["single_result"]["image"]
    assert result.shape == app.session_state["decoded_image"].shape
    assert not np.array_equal(result, app.session_state["decoded_image"])
    assert any("Image comparison" in item.value for item in app.markdown)
    assert any("Export results" in item.value for item in app.markdown)
    assert not app.exception


def test_comparison_and_analytics_render_actual_interactive_charts():
    app = app_with_image("Compare Methods")
    next(button for button in app.button if button.label == "Run comparison").click().run(timeout=60)
    assert len(app.session_state["comparison_results"]["results"]) == 3
    assert len(app.get("plotly_chart")) == 1
    assert not app.exception

    app.radio[0].set_value("Quality Analytics").run(timeout=45)
    next(button for button in app.button if button.label == "Run analytics").click().run(timeout=60)
    assert len(app.session_state["analytics_result"]["rows"]) == 2
    assert len(app.get("plotly_chart")) == 2
    assert not app.exception


def test_batch_processing_page_renders_without_uploads():
    app = app_with_image("Batch Processing")
    assert any("Batch processing" in item.value for item in app.markdown)
    assert any(button.label == "Start processing" for button in app.button)
    assert not app.exception


def test_clear_image_action_removes_cached_source_and_single_result():
    app = app_with_image("Enhance Image")
    app.session_state["single_result"] = {
        "key": None,
        "image": app.session_state["decoded_image"].copy(),
    }
    app.run(timeout=45)
    next(
        button
        for button in app.button
        if button.label == "Clear image and derived results"
    ).click().run(timeout=45)

    assert "decoded_image" not in app.session_state
    assert "single_result" not in app.session_state
    assert not app.exception
