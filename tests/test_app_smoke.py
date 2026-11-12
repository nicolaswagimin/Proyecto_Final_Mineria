"""Prueba de humo de la interfaz: cada sección renderiza sin excepciones.

Cubre errores de figura (por ejemplo, pasar `opacity` a `update_layout`, que es
una propiedad de traza) que solo aparecen al dibujar la vista.
"""

from __future__ import annotations

import pytest

from campaign_optimizer import config

pytestmark = pytest.mark.skipif(
    not config.RAW_CSV.exists() or not config.TRAIN_REPORT.exists(),
    reason="requiere el CSV y los artefactos de entrenamiento",
)

SECTIONS = ["Resumen", "Segmentos", "Modelos", "Recomendaciones"]


@pytest.mark.parametrize("section", SECTIONS)
def test_cada_seccion_renderiza(section):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(config.PROJECT_ROOT / "app.py"), default_timeout=180)
    at.run()
    assert at.exception == [], f"la carga inicial falló: {at.exception}"

    at.radio[0].set_value(section)
    at.run()
    assert at.exception == [], f"{section} lanzó: {[str(e) for e in at.exception]}"
