"""Punto de entrada para Streamlit.

    streamlit run app.py

Mantener el entrypoint en la raíz permite desplegar en Streamlit Community Cloud
sin configurar un paquete instalable.
"""

from campaign_optimizer.app import main

main()
