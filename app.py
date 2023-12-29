import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from streamlit_extras import F
from streamlit_option_menu import option_menu
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_timeline import timeline
import numpy as np
import math
import os
import plotly.graph_objects as go
import scipy.stats as stats
from scipy.stats import t
from modules import Chart, Info, Regression, Classification, Clustering

st.set_page_config("DataApp",layout="wide",initial_sidebar_state="expanded",)


def footer():
    st.markdown(
        """
        <head>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        </head>
        <style>
            footer
            {
            visibility:hidden;
            }
            .a {
                
                background-color: #f0f2f6;
                padding: 20px;
                text-align: center;
            }
            
            .icon-list {
                display: flex;
                justify-content: center;
                align-items: center;
            }

            .icon-list-item {
                margin: 10px;
                text-align: center;
                cursor: pointer;
            }

            .icon-list-item i {
                display: block;
                font-size: 20px;
                color: black;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
    """
        <div class="a">
            <h6>Liên hệ với tôi</h6>
            <div class="icon-list">
                <div class="icon-list-item">
                    <a href="https://github.com" target="_blank">
                        <i class="fab fa-github"></i>
                    </a>
                </div>
                <div class="icon-list-item">
                    <a href="https://twitter.com" target="_blank">
                        <i class="fab fa-twitter"></i>
                    </a>
                </div>
                <div class="icon-list-item">
                    <a href="https://youtube.com" target="_blank">
                        <i class="fab fa-youtube"></i>
                    </a>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
)

def search():
    st.markdown("""
            <head>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
            <style>
                body {
                font-family: Sans serif;
                }

                * {
                box-sizing: border-box;
                }

                form.example input[type=text] {
                padding: 10px;
                font-size: 17px;
                border: 2px solid white;
                float: left;
                width: 700px;  /* Set width to 100px */
                background: #f1f1f1;
                border-radius: 15px;
                }

                form.example button {
                float: left;
                width: 100px;  /* Set width to auto to adjust based on content */
                padding: 10px;
                background: #FF4B4B;
                color: white;
                font-size: 17px;
                border: 2px solid white;
                border-left: none;
                cursor: pointer;
                border-radius: 15px;
                }

                form.example button:hover {
                background: #FF4B4B;
                }

                form.example::after {
                content: "";
                clear: both;
                display: table;
                }
            </style>
            </head>
            <body>
            """, 
            unsafe_allow_html=True)

    st.markdown("""
            <form class="example" action="" style="margin:auto;max-width:800px">
            <input type="text" placeholder="Search.." name="search2">
            <button type="submit"><i class="fa fa-search"></i></button>
            </form>""",
            unsafe_allow_html=True)
    st.markdown(
                """
                <head>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
                </head>
                """,
                unsafe_allow_html=True
            )


    

@st.cache_data
def load_data(file):
    file_extension = os.path.splitext(file.name)[1].lower()
    if file_extension == '.csv':
        return pd.read_csv(file)
    elif file_extension in ['.xlsx', '.xls']:
        return pd.read_excel(file)



def summary(df):
    summary = df.describe()
    var_row = pd.Series(data=df.var(), name='var')
    var_row.index = summary.columns
    summary = pd.concat([summary.loc[['count', 'mean'], :], var_row.to_frame().T, summary.loc[['std', 'min', '25%', '50%', '75%', 'max'], :]])
    summary.loc['skew'] = df.skew()
    return summary

def summary_p(df):
    n = df.shape[0]
    summary = df.describe()
    var_row = pd.Series(data=df.var(), name='var')*((n-1)/n)
    var_row.index = summary.columns
    summary = pd.concat([summary.loc[['count', 'mean'], :], var_row.to_frame().T, summary.loc[['std', 'min', '25%', '50%', '75%', 'max'], :]])
    summary.loc['std'] = df.std()*math.sqrt((n-1)/n)
    summary.loc['skew'] = df.skew()*((n-2)/math.sqrt(n*(n-1)))
    return summary

   




### analyze_data      
def analyze_data(data):
    # Perform basic data analysis
    st.write(" # Data Analysis # ")
    st.write("#### Dữ liệu ####")
    st.write("Data")
    with st.expander("See explanation"):
        edited_df = st.experimental_data_editor(data,use_container_width=True,num_rows="dynamic")
    st.markdown("---")
    ######
    st.write("#### Thống kê mô tả một chiều ####")

    st.markdown("###### Bảng giá trị thống kê mô tả ######")
    use_sample_stats = st.checkbox('Hiệu chỉnh mẫu thống kê', value=True)
    if use_sample_stats:
    # compute and show the sample statistics
        st.dataframe(summary(edited_df),use_container_width=True)
        st.download_button(
        label="Download data as CSV",
        data=summary(data).to_csv(index=False),
        file_name='data_analyze.csv',
        mime='text/csv')
   
    else:
    # compute and show the population statistics
        st.dataframe(summary_p(edited_df),use_container_width=True)
        st.download_button(
        label="Download data as CSV",
        data=summary_p(data).to_csv(index=False),
        file_name='data_analyze.csv',
        mime='text/csv')
    footer()


#### Data viusualyzation




#### hypothesis test
def hypothesis_test(test_type, data):
    # Perform basic data analysis
    
    ######
    if test_type =="Kiểm định một mẫu":
        st.markdown("---")
        st.write("#### Chọn phương thức kiểm định một mẫu mong muốn ####")
        test_type_one = st.selectbox("", ["Kiểm định về giá trị trung bình", "Kiểm định về phương sai"])
        if test_type_one=="Kiểm định về giá trị trung bình":
            st.markdown("---")
            st.write("#### Kiểm định về giá trị trung bình ####")
            st.markdown(
                """
                <style>
                .c {
                    margin-top: 30px ;
                    }
                </style>

                <div class="c"></div>
                """,
                unsafe_allow_html=True
            )
            st.write("##### Chọn cột cần kiểm định #####")
            numeric_columns = data.select_dtypes(include=["int", "float"]).columns
            x_column = st.selectbox("", numeric_columns)
            stats_df = pd.DataFrame({
                "Mean": [data[x_column].mean()],
                "Standard Deviation": [data[x_column].std()],
                "Count": [data[x_column].count()]
            })
                
            st.markdown("Giá trị thống kê tính được")
            reset_df=stats_df.set_index("Mean",drop=True)
            st.dataframe(reset_df,use_st_width=True)

            st.markdown("Các yếu tố: ")
            col1, col2, col3 = st.columns(3)
            with col1:
                clevel = st.text_input('Mức ý nghĩa', '0.05')
            with col2:
                a0 = st.text_input('Giá trị trung bình cần kiểm định', '')

            with col3:
                H1 = st.selectbox("Đối thuyết", ["Khác", "Lớn hơn", "Nhỏ hơn"])
            
            sample = data[x_column].values
            alpha = float(clevel)
            st.markdown("---")   
            

            if a0.strip():  # Check if a0 is not empty or whitespace
                st.markdown("###### Bài toán kiểm định giả thuyết:")
                col1, col2, col3 = st.columns(3)
                with col2:
                    if H1 == "Khác":
                        st.latex(r'''
                    \left\{
                    \begin{aligned}
                        H_0 &: \mu = \mu_0 \\
                        H_1 &: \mu \neq \mu_0
                    \end{aligned}
                    \right.
                    ''')
                    elif H1 == "Lớn hơn":
                        st.latex(r'''
                        \left\{
                        \begin{aligned}
                        H_0 &: \mu = \mu_0 \\
                        H_1 &: \mu > \mu_0
                    \end{aligned}
                        \right.
                        ''')
                    else:
                        st.latex(r'''
                        \left\{
                        \begin{aligned}
                        H_0 &: \mu = \mu_0 \\
                        H_1 &: \mu < \mu_0
                    \end{aligned}
                        \right.
                        ''')
                a0_value = float(a0)
                st.markdown("Thống kê phù hợp t:")
                st.latex(r'''
                t=\dfrac{(\overline{x}-\mu)\sqrt{n}}{s_d}
                ''')
                st.latex(r'''\text{Ta có: }
                t \sim t_{n-1}
                ''')

                if H1 == "Khác":
                    t_statistic, p_value= stats.ttest_1samp(sample, popmean=a0_value)
                    st.markdown(f"Giá trị $$t$$ tính được là: <span style='color: green'> $$t = {t_statistic}$$</span>", unsafe_allow_html=True)
                    percent=stats.t.ppf(q=1-alpha/2, df=data[x_column].count()-1)
                    t_critical_1 = t.ppf(alpha / 2, data[x_column].count()-1)
                    t_critical_2 = t.ppf(1 - alpha / 2, data[x_column].count()-1)

                    # Generate x values for the PDF plot
                    x = np.linspace(-5, 5, 1000)

                    # Calculate the PDF values
                    pdf = t.pdf(x, data[x_column].count()-1)

                    # Plot the PDF
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x, y=pdf, name="PDF"))
                    fig.update_layout(
                        title=f"Student's t-Distribution PDF (df={data[x_column].count()-1})",
                        xaxis_title="x",
                        yaxis_title="PDF",
                    )

                    x_fill1 = np.linspace(-5, t_critical_1, 1000)
                    pdf_fill1 = t.pdf(x_fill1, data[x_column].count()-1)

                    x_fill2 = np.linspace(t_critical_2, 5, 1000)
                    pdf_fill2 = t.pdf(x_fill2, data[x_column].count()-1)

                    # Highlight the area under the curve    
                    fig.add_trace(go.Scatter(x=x_fill1, y=pdf_fill1, fill='tozeroy', fillcolor='rgba(100, 10, 10, 0.3)',
                                mode='lines', line=dict(color='rgba(0, 0, 0, 0)'), name='Area Under Curve'))
                    fig.add_trace(go.Scatter(x=x_fill2, y=pdf_fill2, fill='tozeroy', fillcolor='rgba(100, 10, 10, 0.3)',
                                mode='lines', line=dict(color='rgba(0, 0, 0, 0)'), name='Area Under Curve'))

                    # Highlight the two tail areas
                    fig.add_trace(go.Scatter(x=[t_critical_1, t_critical_1], y=[0, t.pdf(t_critical_1, data[x_column].count()-1)],
                                mode="lines", name="Left Tail Area", line=dict(color="red", dash="dash")))
                    fig.add_trace(go.Scatter(x=[t_critical_2, t_critical_2], y=[0, t.pdf(t_critical_2, data[x_column].count()-1)],
                                mode="lines", name="Right Tail Area", line=dict(color="red", dash="dash")))

                    # Display the plot
                    st.plotly_chart(fig,theme=None, use_st_width=True)
                    
                    

                    inf = r"\infty"
                    hop = r"\cup"
                    st.markdown("##### Kết luận")
                    st.markdown(f"Miền bác bỏ hai phía ($$-{inf},{t_critical_1}$$) $${hop}$$ ($${t_critical_2},{inf}$$)")
                    if(np.abs(t_statistic) > percent):
                        latex_expression = r"t_{n-1}(\frac{\alpha}{2})"
                        st.markdown(f"Vì |t_statistic| = :green[{np.abs(t_statistic)}] > $$ {latex_expression}$$ = :green[{percent}] ")
                        st.markdown(f"Nên ta bác bỏ giả thuyết H0 ở mức ý nghĩa :green[{alpha}]")
                    else:
                        latex_expression = r"t_{n-1}(\frac{\alpha}{2})"
                        st.markdown(f"Vì |t_statistic|= :green[{np.abs(t_statistic)}] < $$ {latex_expression}$$=:green[{percent}] ")
                        st.markdown(f"Nên ta chấp nhận giả thuyết H0 ở mức ý nghĩa :green[{alpha}]")

                elif H1 == "Lớn hơn":
                    percent=stats.t.ppf(q=1-alpha, df=data[x_column].count()-1)
                    t_statistic = (data[x_column].mean() - a0_value) / (data[x_column].std() / math.sqrt(data[x_column].count()))
                    st.markdown(f"t-statistic= :green[{t_statistic}]")
                    t_critical = stats.t.ppf(1 - alpha, df=data[x_column].count()-1)

                    # Generate x values for the PDF plot
                    x = np.linspace(-5, 5, 1000)

                    # Calculate the PDF values
                    pdf = stats.t.pdf(x, df=data[x_column].count()-1)

                    # Plot the PDF
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x, y=pdf, name="PDF"))
                    fig.update_layout(
                    title=f"Student's t-Distribution PDF (df={data[x_column].count()-1})",
                    xaxis_title="x",
                    yaxis_title="PDF",
                    )

                    x_fill = np.linspace(t_critical, x[-1], 1000)
                    pdf_fill = stats.t.pdf(x_fill, df=data[x_column].count()-1)

                    # Highlight the area under the curve
                    fig.add_trace(go.Scatter(x=x_fill, y=pdf_fill, fill='tozeroy', fillcolor='rgba(100, 10, 10, 0.3)',
                                mode='lines', line=dict(color='rgba(0, 0, 0, 0)'), name='Area Under Curve'))

                    # Highlight the critical region
                    fig.add_trace(go.Scatter(x=[t_critical, t_critical], y=[0, stats.t.pdf(t_critical, df=data[x_column].count()-1)],
                                mode="lines", name="Critical Region", line=dict(color="red", dash="dash")))


                    st.plotly_chart(fig, theme=None, use_st_width=True)
                    inf = r"\infty"
                    hop = r"\cup"
                    st.markdown("##### Kết luận")
                    st.markdown(f"Miền bác bỏ một phía ($$-{inf},{t_critical}$$) ")
                    if(t_statistic > percent):
                        latex_expression = r"t_{n-1}({1- \alpha})"
                        st.markdown(f"Vì t_statistic= :green[{t_statistic}] > $$ {latex_expression}$$ = :green[{percent}] ")
                        st.markdown(f"nên ta bác bỏ giả thuyết H0 ở mức ý nghĩa :green[{alpha}]")
                    else:
                        latex_expression = r"t_{n-1}({1- \alpha})"
                        st.markdown(f"Vì t_statistic= :green[{t_statistic}] < $$ {latex_expression}$$=:green[{percent}] ")
                        st.markdown(f"nên ta chấp nhận giả thuyết H0 ở mức ý nghĩa :green[{alpha}]")  
                else:
                    percent=stats.t.ppf(q=alpha, df=data[x_column].count()-1)
                    t_statistic = (data[x_column].mean() - a0_value) / (data[x_column].std() / math.sqrt(data[x_column].count()))
                    st.markdown(f"t-statistic= :green[{t_statistic}]")
                    t_critical = stats.t.ppf(alpha, df=data[x_column].count()-1)
                    # Generate x values for the PDF plot
                    x = np.linspace(-5, 5, 1000)

                    # Calculate the PDF values
                    pdf = stats.t.pdf(x, df=data[x_column].count()-1)

                    # Plot the PDF
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x, y=pdf, name="PDF"))
                    fig.update_layout(
                    title=f"Student's t-Distribution PDF (df={data[x_column].count()-1})",
                    xaxis_title="x",
                    yaxis_title="PDF",
                    )

                    x_fill = np.linspace(-5,t_critical, 1000)
                    pdf_fill = stats.t.pdf(x_fill, df=data[x_column].count()-1)

                    # Highlight the area under the curve
                    fig.add_trace(go.Scatter(x=x_fill, y=pdf_fill, fill='tozeroy', fillcolor='rgba(100, 10, 10, 0.3)',
                                mode='lines', line=dict(color='rgba(0, 0, 0, 0)'), name='Area Under Curve'))

                    # Highlight the critical region
                    fig.add_trace(go.Scatter(x=[t_critical, t_critical], y=[0, stats.t.pdf(t_critical, df=data[x_column].count()-1)],
                                mode="lines", name="Critical Region", line=dict(color="red", dash="dash")))

                    st.plotly_chart(fig, theme=None, use_st_width=True)
                    inf = r"\infty"
                    hop = r"\cup"
                    st.markdown("##### Kết luận")
                    st.markdown(f"Miền bác bỏ một phía ($${t_critical},{inf}$$) ")
                    if(t_statistic < percent):
                        latex_expression = r"t_{n-1}({\alpha})"
                        st.markdown(f"Vì t_statistic= :green[{t_statistic}] < $$ {latex_expression}$$ = :green[{percent}] ")
                        st.markdown(f"nên ta bác bỏ giả thuyết H0 ở mức ý nghĩa :green[{alpha}]")
                    else:
                        latex_expression = r"t_{n-1}({\alpha})"
                        st.markdown(f"Vì t_statistic= :green[{t_statistic}] > $$ {latex_expression}$$=:green[{percent}] ")
                        st.markdown(f"nên ta chấp nhận giả thuyết H0 ở mức ý nghĩa :green[{alpha}]")  

        #
        if test_type_one=="Kiểm định về phương sai":
            st.write("#### Kiểm định về phương sai ####")
            numeric_columns = data.select_dtypes(include=["int", "float"]).columns
            x_column = st.selectbox("Chọn cột cần kiểm định ", numeric_columns)
            stats_df = pd.DataFrame({
                    "Variance": [data[x_column].var()],
                    "Count": [data[x_column].count()]
                })
                    
            st.markdown("Giá trị thống kê tính được")
            reset_df = stats_df.set_index("Variance", drop=True)
            st.dataframe(reset_df, use_st_width=True)
            st.markdown("Các yếu tố: ")
            col1, col2 = st.columns(2)
            with col1:
                clevel = st.text_input('Mức ý nghĩa', '0.05')
            with col2:
                a0 = st.text_input('Giá trị phương sai cần kiểm định', '')


            sample = data[x_column].values
            alpha = float(clevel)
            st.markdown("---")   

            if a0.strip():  # Check if a0 is not empty or whitespace
                st.markdown("###### Bài toán kiểm định giả thuyết:")
                col1, col2,col3 = st.columns(3)
                with col2:
                    st.latex(r'''
                    \left\{
                    \begin{aligned}
                        H_0 &: \sigma^2 = \sigma_0^2 \\
                        H_1 &: \sigma^2 \neq \sigma_0^2
                    \end{aligned}
                    \right.
                    ''')

                

                a0_value = float(a0)
                st.markdown("Thống kê phù hợp chi-square:")
                st.latex(r'''
                \chi^2 = (n-1) \cdot \frac{{s^2}}{{\sigma_0^2}}
                ''')
                st.latex(r'''\text{Ta có: }
                \chi^2 \sim \chi^2_{n-1}
                ''')
                
                
                chi2_statistic = (data[x_column].count() - 1) * data[x_column].var() / a0_value
                st.markdown(f"chi-square statistic = :green[{chi2_statistic}]")
                chi2_critical = stats.chi2.ppf(1 - alpha / 2, df=data[x_column].count() - 1)
                chi2_critical2 = stats.chi2.ppf(alpha / 2, df=data[x_column].count() - 1)

                # Generate x values for the Chi-square distribution plot
                x = np.linspace(stats.chi2.ppf(alpha / 2, df=data[x_column].count() - 1)-20, stats.chi2.ppf(1 - alpha / 2, df=data[x_column].count() - 1)+20, 1000)

                # Calculate the PDF values
                pdf = stats.chi2.pdf(x, df=data[x_column].count()-1)

                # Plot the PDF
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=pdf, name="PDF"))
                fig.update_layout(
                    title=f"Chi-square Distribution PDF (df={data[x_column].count()-1})",
                    xaxis_title="x",
                    yaxis_title="PDF",
                )

                x_fill1 = np.linspace(x[0], chi2_critical2, 1000)
                pdf_fill1 = stats.chi2.pdf(x_fill1, df=data[x_column].count()-1)

                x_fill2 = np.linspace(chi2_critical,x[-1], 1000)
                pdf_fill2 = stats.chi2.pdf(x_fill2, df=data[x_column].count()-1)

                fig.add_trace(go.Scatter(x=x_fill1, y=pdf_fill1, fill='tozeroy', fillcolor='rgba(100, 10, 10, 0.3)',
                                mode='lines', line=dict(color='rgba(0, 0, 0, 0)'), name='Area Under Curve'))
                fig.add_trace(go.Scatter(x=x_fill2, y=pdf_fill2, fill='tozeroy', fillcolor='rgba(100, 10, 10, 0.3)',
                                mode='lines', line=dict(color='rgba(0, 0, 0, 0)'), name='Area Under Curve'))
                
                fig.add_trace(go.Scatter(x=[chi2_critical, chi2_critical], y=[0, stats.chi2.pdf(chi2_critical, data[x_column].count()-1)],
                                mode="lines", name="Left Tail Area", line=dict(color="red", dash="dash")))
                fig.add_trace(go.Scatter(x=[chi2_critical2, chi2_critical2], y=[0, stats.chi2.pdf(chi2_critical2, data[x_column].count()-1)],
                                mode="lines", name="Right Tail Area", line=dict(color="red", dash="dash")))
                
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig,theme=None, use_st_width=True)
                
                if chi2_statistic > chi2_critical or chi2_statistic < chi2_critical2:
                    st.markdown(":red[Không chấp nhận null hypothesis]")
                    st.markdown("Có bằng chứng đủ để bác bỏ giả thuyết H0.")
                else:
                    st.markdown(":green[Chấp nhận null hypothesis]")
                    st.markdown("Không có bằng chứng đủ để bác bỏ giả thuyết H0.")
    if test_type =="Kiểm định nhiều mẫu":
        st.markdown("---")
        st.write("#### Chọn phương thức kiểm định nhiều mẫu mong muốn ####")
        test_type_two = st.selectbox("", ["So sánh hai giá trị trung bình", "So sánh hai phương sai", "Phân tích phương sai"])
        if test_type_two=="So sánh hai giá trị trung bình":
            st.markdown("---")
            st.write("#### So sánh hai giá trị trung bình ####")
            st.markdown(
                """
                <style>
                .c {
                    margin-top: 30px ;
                    }
                </style>

                <div class="c"></div>
                """,
                unsafe_allow_html=True
            )
            st.write("##### Chọn các cột cần so sánh #####")
            numeric_columns = data.select_dtypes(include=["int", "float"]).columns
            col1, col2  = st.columns(2)
            with col1:
                x_column = st.selectbox("Mẫu 1", numeric_columns)
            with col2:
                y_column = st.selectbox("Mẫu 2", numeric_columns)
            stats_df = pd.DataFrame({
                "Mẫu 1": [data[x_column].mean(), data[x_column].std(), data[x_column].count()],
                "Mẫu 2": [data[y_column].mean(), data[y_column].std(), data[y_column].count()]
            }, index=["Mean", "Standard Deviation", "Count"])
                
            st.markdown("Giá trị thống kê tính được")
            st.dataframe(stats_df, use_st_width=True)

            st.markdown("Các yếu tố: ")
            col1, col2  = st.columns(2)
            with col1:
                clevel = st.text_input('Mức ý nghĩa', '0.05')
            with col2:
                a0 = st.text_input('Giá trị cần so sánh', '')
            
            
            alpha = float(clevel)
            st.markdown("---")   
            

            if a0.strip():  # Check if a0 is not empty or whitespace
                st.markdown("###### Bài toán kiểm định giả thuyết:")
                col1, col2, col3 = st.columns(3)
                with col2:
                        st.latex(r'''
                        \left\{
                        \begin{aligned}
                        H_0 &: \mu_1 - \mu_2   = a_0 \\
                        H_1 &: \mu_1 - \mu_2   \neq a_0
                    \end{aligned}
                        \right.
                        ''')
                
                a0_value = float(a0)
                st.markdown("Thống kê phù hợp t:")
                if data[x_column].count() > 30:
                    st.latex(r'''
                    t=\frac{{\overline{x_1} - \overline{x_2}-(\mu_1 -\mu_2)}}{\sqrt{\frac{{s_1^2}}{{n_1}}+\frac{s_2^2}{{n_2}}}}
                    ''')
                    st.latex(r'''\text{Ta có: }
                    t \sim t_{min{(n_1-1, n_2-1)}}
                    ''')
                else:
                    st.latex(r'''
                    t=\frac{{\overline{x_1} - \overline{x_2}-(\mu_1 -\mu_2)}}{{\sqrt{\frac{{(n_1-1)s_1^2+(n_2-1)s_2^2}}{{n_1+n_2-2}}(\frac{{1}}{{n_1}}+\frac{1}{{n_2}})}}}
                    ''')
                    st.latex(r'''\text{Ta có: }
                    t \sim t_{n_1+n_2-2}
                    ''')
                t_statistic2 = (data[x_column].mean() - data[y_column].mean() - a0_value) / (math.sqrt((((data[x_column].count()-1)*data[x_column].var()+(data[y_column].count()-1)*data[y_column].var())/((data[x_column].count()+data[y_column].count()-2)))*(1/(data[x_column].count())+1/data[y_column].count())))
                st.markdown(f"t statistic = :green[{t_statistic2}]")
                t_critical_1 = t.ppf(alpha / 2, data[x_column].count()-1)
                t_critical_2 = t.ppf(1 - alpha / 2, data[x_column].count()-1)

                    # Generate x values for the PDF plot
                x = np.linspace(-5, 5, 1000)

                    # Calculate the PDF values
                pdf = t.pdf(x, data[x_column].count()+data[y_column].count()-2)

                    # Plot the PDF
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=pdf, name="PDF"))
                fig.update_layout(
                        title=f"Student's t-Distribution PDF (df={data[x_column].count()+data[y_column].count()-2})",
                        xaxis_title="x",
                        yaxis_title="PDF",
                    )

                x_fill1 = np.linspace(-5, t_critical_1, 1000)
                pdf_fill1 = t.pdf(x_fill1, data[x_column].count()+data[y_column].count()-2)

                x_fill2 = np.linspace(t_critical_2, 5, 1000)
                pdf_fill2 = t.pdf(x_fill2, data[x_column].count()+data[y_column].count()-2)

                    # Highlight the area under the curve    
                fig.add_trace(go.Scatter(x=x_fill1, y=pdf_fill1, fill='tozeroy', fillcolor='rgba(100, 10, 10, 0.3)',
                                mode='lines', line=dict(color='rgba(0, 0, 0, 0)'), name='Area Under Curve'))
                fig.add_trace(go.Scatter(x=x_fill2, y=pdf_fill2, fill='tozeroy', fillcolor='rgba(100, 10, 10, 0.3)',
                                mode='lines', line=dict(color='rgba(0, 0, 0, 0)'), name='Area Under Curve'))

                    # Highlight the two tail areas
                fig.add_trace(go.Scatter(x=[t_critical_1, t_critical_1], y=[0, t.pdf(t_critical_1, data[x_column].count()+data[y_column].count()-2)],
                            mode="lines", name="Left Tail Area", line=dict(color="red", dash="dash")))
                fig.add_trace(go.Scatter(x=[t_critical_2, t_critical_2], y=[0, t.pdf(t_critical_2, data[x_column].count()+data[y_column].count()-2)],
                                mode="lines", name="Right Tail Area", line=dict(color="red", dash="dash")))

                    # Display the plot
                st.plotly_chart(fig,theme=None, use_st_width=True)
        if test_type_two =="So sánh hai phương sai":
            st.write("#### So sánh hai phương sai ####")
            numeric_columns = data.select_dtypes(include=["int", "float"]).columns
            col1,col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("Mẫu 1 ", numeric_columns)
            with col2:
                y_column = st.selectbox("Mẫu 2", numeric_columns)
            stats_df = pd.DataFrame({
                "Mẫu 1": [ data[x_column].var(), data[x_column].count()],
                "Mẫu 2": [ data[y_column].var(), data[y_column].count()]
            }, index=["Variance", "Count"])

            st.markdown("Giá trị thống kê tính được")
            st.dataframe(stats_df, use_st_width=True)

            st.markdown("Các yếu tố: ")
            col1, col2 = st.columns(2)
            with col1:
                clevel = st.text_input('Mức ý nghĩa', '0.05')
            with col2:
                H1 = st.selectbox("Đối thuyết", ["Khác", "Lớn hơn", "Nhỏ hơn"])


            sample = data[x_column].values
            alpha = float(clevel)
            st.markdown("---")   

            if H1=="Khác":  # Check if a0 is not empty or whitespace
                st.markdown("###### Bài toán kiểm định giả thuyết:")
                col1, col2,col3 = st.columns(3)
                with col2:
                    st.latex(r'''
                    \left\{
                    \begin{aligned}
                        H_0 &: \sigma_1^2 = \sigma_2^2 \\
                        H_1 &: \sigma_1^2 \neq \sigma_2^2
                    \end{aligned}
                    \right.
                    ''')
                
                st.markdown("Thống kê phù hợp chi-square:")
                st.latex(r'''
                F = \frac{{s_1^2 \sigma_2^2}}{{s_2^2 \sigma_1^2}}
                ''')
                st.latex(r'''\text{Ta có: }
                F \sim F_{n_1-1,n_2-1}
                ''')
                
                
                F_statistic = (data[x_column].var())/ (data[y_column].var())
                st.markdown(f"chi-square statistic = :green[{F_statistic}]")
                F_critical = stats.f.ppf(1 - alpha / 2, data[x_column].count() - 1,data[y_column].count() - 1)
                F_critical2 = stats.f.ppf(alpha / 2, data[x_column].count() - 1,data[y_column].count() - 1)

                # Generate x values for the Chi-square distribution plot
                x = np.linspace(-2*F_critical2, 2*F_critical, 1000)

                # Calculate the PDF values
                pdf = stats.f.pdf(x, data[x_column].count() - 1,data[y_column].count() - 1)

                # Plot the PDF
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=pdf, name="PDF"))
                fig.update_layout(
                    title=f"F Distribution PDF (df={data[x_column].count()-1},{data[y_column].count()-1})",
                    xaxis_title="x",
                    yaxis_title="PDF",
                )

                x_fill1 = np.linspace(x[0], F_critical2, 1000)
                pdf_fill1 = stats.f.pdf(x_fill1, data[x_column].count() - 1,data[y_column].count() - 1)

                x_fill2 = np.linspace(F_critical,x[-1], 1000)
                pdf_fill2 = stats.f.pdf(x_fill2, data[x_column].count() - 1,data[y_column].count() - 1)

                fig.add_trace(go.Scatter(x=x_fill1, y=pdf_fill1, fill='tozeroy', fillcolor='rgba(100, 10, 10, 0.3)',
                                mode='lines', line=dict(color='rgba(0, 0, 0, 0)'), name='Area Under Curve'))
                fig.add_trace(go.Scatter(x=x_fill2, y=pdf_fill2, fill='tozeroy', fillcolor='rgba(100, 10, 10, 0.3)',
                                mode='lines', line=dict(color='rgba(0, 0, 0, 0)'), name='Area Under Curve'))
                
                fig.add_trace(go.Scatter(x=[F_critical, F_critical], y=[0, stats.f.pdf(F_critical, data[x_column].count() - 1,data[y_column].count() - 1)],
                                mode="lines", name="Left Tail Area", line=dict(color="red", dash="dash")))
                fig.add_trace(go.Scatter(x=[F_critical2, F_critical2], y=[0, stats.f.pdf(F_critical2, data[x_column].count() - 1,data[y_column].count() - 1)],
                                mode="lines", name="Right Tail Area", line=dict(color="red", dash="dash")))
                
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig,theme=None, use_st_width=True)

        if test_type_two =="Phân tích phương sai":
            st.write("#### ANOVA Test ####")

            st.write("##### Chọn các cột cho ANOVA #####")
            numeric_columns = data.select_dtypes(include=["int", "float"]).columns
            col1,col2 = st.columns([3,1])
            with col1:
                columns = st.multiselect("Select columns for ANOVA test", numeric_columns)
            with col2:
                clevel = st.text_input('Mức ý nghĩa', '0.05')
            alpha = float(clevel)
            if len(columns) > 1:
    # Filter the DataFrame to include only the selected columns
                selected_data = data[columns]

                # Convert each column to a Series and drop NaN values
                series_data = []
                for col in columns:
                    series = selected_data[col].dropna()
                    series_data.append(series)
                
                if len(series_data) > 1:
                    # Perform the ANOVA analysis
                    summary_table = pd.DataFrame(columns=["Column", "Mean", "Standard Deviation", "Count"])
                    for i, col in enumerate(columns):
                        summary_table = summary_table.append({
                            "Column": col,
                            "Mean": series_data[i].mean(),
                            "Standard Deviation": series_data[i].std(),
                            "Count": series_data[i].count()
                        }, ignore_index=True)
                    # Display the column summaries in Streamlit
                    st.write("#### Giá trị thống kê tính được ####")
                    st.dataframe(summary_table, use_st_width=True)
                    f_statistic, p_value = stats.f_oneway(*series_data)
                    within_squares = sum(sum((series - series.mean())**2) for series in series_data)
                    overall_mean = pd.Series(selected_data.values.flatten()).dropna().mean()
                    between_squares = sum(len(series) * (series.mean() - overall_mean)**2 for series in series_data)
                    total_squares = within_squares + between_squares
                        
                    # Create the ANOVA table
                    anova_table = pd.DataFrame({
                        "Nguồn biến thiên": ["Xử lý", "Phần dư", "Tổng cộng"],
                        "Tổng bình phương": [
                            between_squares,
                            within_squares,
                            total_squares
                        ],
                            
                        "Bậc tự do": [
                            len(series_data) - 1,
                            sum(len(series) - 1 for series in series_data),
                            sum(len(series) for series in series_data) - 1
                        ],
                        "Tỉ số MS": [
                            between_squares / (len(series_data) - 1),
                            within_squares / sum(len(series) - 1 for series in series_data),
                            ""
                        ],
                        "F-Value": [
                            "",
                            f_statistic,
                            ""
                        ],
                        "p-value": [
                            "",
                            p_value,
                            ""
                        ]
                    })

                    # Display the ANOVA table in Streamlit
                    st.write("#### Bảng ANOVA ####")
                    st.dataframe(anova_table, use_st_width=True)

                    F_critical = stats.f.ppf(1 - alpha , (len(series_data) - 1),sum(len(series) - 1 for series in series_data))
                    

                    # Generate x values for the Chi-square distribution plot
                    x = np.linspace(0, 2*F_critical, 1000)

                    # Calculate the PDF values
                    pdf = stats.f.pdf(x, (len(series_data) - 1),sum(len(series) - 1 for series in series_data))

                    # Plot the PDF
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x, y=pdf, name="PDF"))
                    fig.update_layout(
                        title=f"F Distribution PDF (df={(len(series_data) - 1)},{sum(len(series) - 1 for series in series_data)})",
                        xaxis_title="x",
                        yaxis_title="PDF",
                    )

                    
                    x_fill2 = np.linspace(F_critical,x[-1], 1000)
                    pdf_fill2 = stats.f.pdf(x_fill2, (len(series_data) - 1),sum(len(series) - 1 for series in series_data))


                    fig.add_trace(go.Scatter(x=x_fill2, y=pdf_fill2, fill='tozeroy', fillcolor='rgba(100, 10, 10, 0.3)',
                                    mode='lines', line=dict(color='rgba(0, 0, 0, 0)'), name='Area Under Curve'))
                    
                    fig.add_trace(go.Scatter(x=[F_critical, F_critical], y=[0, stats.f.pdf(F_critical, (len(series_data) - 1),sum(len(series) - 1 for series in series_data))],
                                    mode="lines", name="Left Tail Area", line=dict(color="red", dash="dash")))
                    
                    
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig,theme=None, use_st_width=True)
                else:
                    st.write("Insufficient data points after preprocessing. Please select columns with valid numeric values.")
            else:
                st.write("Please select at least two columns for the ANOVA test.")


    if test_type =="Kiểm định phi tham số":
        st.markdown("---")
        st.write("#### Chọn phương thức kiểm định mong muốn ####")
        test_type_three = st.selectbox("", ["Kiểm định phân phối chuẩn"])
        if test_type_three == "Kiểm định phân phối chuẩn":
            numeric_columns = data.select_dtypes(include=["int", "float"]).columns
            column = st.selectbox("Select a Column", numeric_columns)
            sort_col = data[column].sort_values().reset_index(drop=True)

            # Calculate z-scores for the selected column
            z_scores = (sort_col.index + 1 - 0.5) / len(data[column])

            # Generate theoretical quantiles
            quantiles = np.linspace(sort_col[0], sort_col[5], len(data[column]))
            theoretical_quantiles = stats.norm.ppf(z_scores)

            # Create the QQ plot
            qq_fig = go.Figure()
            qq_fig.add_trace(go.Scatter(x=sort_col, y=theoretical_quantiles, mode="markers", name="QQ Plot"))

            # Add linear regression line
            slope, intercept, _, _, _ = stats.linregress(sort_col, theoretical_quantiles)
            regression_line = intercept + slope * sort_col
            qq_fig.add_trace(go.Scatter(x=sort_col, y=regression_line, mode="lines", name="Linear",line=dict(color='red')))

            qq_fig.update_layout(
                title=f"Biểu đồ Q-Q plot",
                xaxis_title="Sample Quantiles",
                yaxis_title="Theoretical Quantiles",
            )
            
            
            

            # Display the QQ plot
            
            
            st.plotly_chart(qq_fig,theme = None, use_st_width=True)
            
            st.write(np.corrcoef(sort_col, theoretical_quantiles))


        
       
# main function
def main():

    with st.sidebar:
        st.sidebar.markdown("---")
        st.markdown("#### Chọn chức năng ####")
        selected = option_menu(None, ["Dữ liệu", "Thống kê", "Trực quan hóa", "Hồi quy", "Phân lớp","Phân cụm"], 
            icons=['clipboard-data', 'table', "bar-chart-fill", 'rulers', 'diamond-half'], 
            menu_icon="cast", default_index=0,styles={
                                   "st": {"padding": "5!important", "background-color": "#fafafa"},
                                   "icon": {"color": "black", "font-size": "15px"},
                                   "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px",
                                                "--hover-color": "#eee"},
                                   })

    with st.container():
        with st.sidebar:
            st.sidebar.markdown("---")
            st.markdown("#### Tải lên dữ liệu ####")
            file = st.file_uploader("",type=["csv", "xlsx", "xls"])

        if file is not None:

            data = load_data(file)

            if selected =='Dữ liệu':
                search()
                Info.info(data)

            if selected == 'Thống kê':
                search()
                analyze_data(data)

            if selected =='Trực quan hóa':
                search()
                st.write(" # Trực quan hóa dữ liệu # ")
                st.write("#### Dữ liệu ####")
                st.write("Data")
                edit_data= st.experimental_data_editor(data,use_container_width=True,num_rows="dynamic")
                st.markdown("---")
                st.write("#### Chọn loại biểu đồ ####")
                chart_type = st.selectbox("", ["Bar"])

                Chart.create_chart(chart_type, edit_data)
                
            if selected == 'Hồi quy':
                search()
                
                st.write(" # Hồi quy tuyến tính # ")
                st.write("#### Dữ liệu ####")
                st.write("Data")
                edit_data= st.experimental_data_editor(data,use_container_width=True,num_rows="dynamic")
                st.markdown("---")
                regression_type = st.selectbox("", ["Hồi quy một biến", 'Hồi quy đa biến'])
                if regression_type =="Hồi quy một biến":
                    Regression.simple_linear_regresstion(data)
                if regression_type == "Hồi quy đa biến":
                    Regression.multiple_linear_regression(data)
                    
            if selected == 'Phân lớp':
                search()
                st.write(" # Phân lớp # ")
                st.write("#### Dữ liệu ####")
                st.write("Data")
                edit_data= st.experimental_data_editor(data,use_container_width=True,num_rows="dynamic")
                st.markdown("---")
                class_type = st.selectbox("", ["KNN", 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'SVM'])
                if class_type == 'KNN':
                    Classification.knn_classification(data)
                if class_type == 'Logistic Regression':
                    Classification.lgreg_classification(data)
                if class_type == 'Random Forest':
                    Classification.randomfor_classification(data)
                if class_type == 'Naive Bayes':
                    Classification.naivebayes_classification(data)
                if class_type == 'SVM':
                    Classification.svm_classification(data)
                
            
            if selected == 'Phân cụm':
                search()
                st.write(" # Phân cụm # ")
                st.write("#### Dữ liệu ####")
                st.write("Data")
                edit_data= st.experimental_data_editor(data,use_container_width=True,num_rows="dynamic")
                st.markdown("---")
                Clustering.kmeans_clustering(data)
                

            # if selected =='Kiểm định':
            #     search()
            #     st.write(" # Kiểm định giả thuyết thống kê # ")
            #     st.write("#### Dữ liệu ####")
            #     st.write("Data")
            #     edit_data= st.experimental_data_editor(data,use_st_width=True,num_rows="dynamic")
            #     st.markdown("---")
            #     st.write("#### Chọn phương thức muốn kiểm định ####")
            #     test_type = st.selectbox("", [None,"Kiểm định một mẫu", "Kiểm định nhiều mẫu", "Kiểm định phi tham số"])
            #     hypothesis_test(test_type, edit_data)
        else:
            st.balloons()
            st.markdown(
        """
        <h1>Trang chủ</h1>
        """,
        unsafe_allow_html=True,
)
            
    
if __name__ == "__main__":
    main()
    

    

    

