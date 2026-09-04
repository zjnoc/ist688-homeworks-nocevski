import streamlit as st

st.set_page_config(page_title="HW Manager")

HW1 = st.Page("HW1.py", title="HW_01")
HW2 = st.Page("HW2.py", title="HW_02", default=True)

pg = st.navigation([HW1, HW2])

pg.run()