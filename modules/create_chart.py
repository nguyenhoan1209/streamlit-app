import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from streamlit_extras import F
from streamlit_option_menu import option_menu
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_timeline import timeline
from streamlit_image_select import image_select
import numpy as np
import math
import os
import plotly.graph_objects as go
import scipy.stats as stats
from scipy.stats import t
from scipy.stats import chi2_contingency
import plotly.figure_factory as ff

class Chart:
    def create_chart(chart_type, data):
        col1,col2 = st.columns(2)
        if chart_type == "Bar":
        
            st.header("Bar Chart")
            with col1:
                x_column = st.selectbox("Chọn trục X", data.columns)
            with col2:
                y_column = st.selectbox("Chọn trục Y", data.columns)
            fig = px.bar(data, x=x_column, y=y_column,color = x_column)
            st.plotly_chart(fig,theme=None, use_container_width=True)

