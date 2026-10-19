import streamlit as st

st.set_page_config(page_title="HW Manager")

HW1 = st.Page("HW1.py", title="HW_01")
HW2 = st.Page("HW2.py", title="HW_02")
HW3 = st.Page("HW3.py", title="HW_03")
HW4 = st.Page("HW4.py", title="HW_04")
HW5 = st.Page("HW5.py", title="HW_05")
HW7 = st.Page("HW7 Folder/HW7.py", title="HW_07", default=True)

pg = st.navigation([HW1, HW2, HW3, HW4, HW5, HW7])

pg.run()