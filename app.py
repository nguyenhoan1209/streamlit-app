import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from streamlit_option_menu import option_menu
from streamlit_extras.dataframe_explorer import dataframe_explorer
import numpy as np
import math
import plotly.graph_objects as go
import scipy.stats as stats
from scipy.stats import t


st.set_page_config("DataApp","📊",layout="wide",initial_sidebar_state="expanded",)

container = st.container()

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


    

@st.cache_data
def load_data(file):
    return pd.read_csv(file)



def sort_data(df):
    
    # Sort Data
    sort_column = st.selectbox("Sort by", df.columns)
    df = df.sort_values(by=sort_column)
    return df

    
    
def group_by(df):
     # Group Data
    group_column = st.selectbox("Group by Sum",df.columns)
    grouped_df = df.groupby(group_column).sum()
    return grouped_df

def group_by_mean(df):
     # Group Data
    group_column = st.selectbox("Group by Mean",df.columns)
    grouped_df_mean = df.groupby(group_column).mean()
    return grouped_df_mean


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

   

def info(data):
    container.write(" # Thông tin dữ liệu # ")
    container.write("#### Dữ liệu ####")
    container.write("Dữ liệu")
    filtered_df = dataframe_explorer(data, case=False)
    container.dataframe(filtered_df, use_container_width=True)
    st.download_button(
            label="Download filter data",
            data=filtered_df.to_csv(index=False),
            file_name='data_filter.csv',
            mime='text/csv',
            )
    container.markdown("---")

    container.write("#### Thông tin ####")
    r=data.shape[0]
    c=data.shape[1]
    container.markdown(f"Kích thước dữ liệu: :red[{r}] x :red[{c}]")
    
    col1,col2,col3 = st.columns(3)

      
    

    with col1:
          
       st.write("Tên cột: ")
       st.dataframe(data.columns,use_container_width=True)
    with col2:
          
       st.write("Kiểu dữ liệu cột: ")
       st.dataframe(data.dtypes,use_container_width=True)
        
    with col3:
       st.write("Unique Values: ")
       st.dataframe(data.nunique(),use_container_width=True)

    container.markdown("---")
    st.markdown("#### Missing Values ####")
    col1n,col2n = st.columns([1,3])
    with col1n:   
        st.dataframe(data.isnull().sum(),use_container_width=True)
    with col2n:
        st.markdown("   Missing value (giá trị thiếu) trong khoa học dữ liệu và phân tích dữ liệu là giá trị không tồn tại hoặc không được xác định trong tập dữ liệu.")
        st.markdown("Missing value có thể xảy ra khi dữ liệu bị mất hoặc bị lỗi trong quá trình thu thập hoặc nhập liệu, hoặc có thể do một giá trị không có ý nghĩa được đại diện bởi các giá trị như NaN (Not a Number) hoặc NULL trong các ngôn ngữ lập trình.")
        data.dropna(inplace=True)
        data.reset_index(drop=True, inplace=True)
        st.download_button(
            label="Download clean data",
            data=data.to_csv(index=False),
            file_name='data_clean.csv',
            mime='text/csv',
            )
        container.markdown("---")
    footer()
    image = Image.open("image/sami.jpg")
    with st.sidebar:
        st.sidebar.image(image,width = 50)
        st.sidebar.markdown("# Data #")
        st.sidebar.markdown("---")
        st.sidebar.markdown("- Data")
        st.sidebar.markdown("- Thông tin")
        st.sidebar.markdown("- Missing Value")


### analyze_data      
def analyze_data(data):
    # Perform basic data analysis
    container.write(" # Data Analysis # ")
    container.write("#### Dữ liệu ####")
    container.write("Data")
    container.dataframe(data,use_container_width=True)
    container.markdown("---")
    ######
    container.write("#### Thống kê mô tả một chiều ####")

    container.markdown("###### Bảng giá trị thống kê mô tả ######")
    use_sample_stats = st.checkbox('Hiệu chỉnh mẫu thống kê', value=True)
    if use_sample_stats:
    # compute and show the sample statistics
        container.dataframe(summary(data),use_container_width=True)
        container.download_button(
        label="Download data as CSV",
        data=summary(data).to_csv(index=False),
        file_name='data_analyze.csv',
        mime='text/csv')
   
    else:
    # compute and show the population statistics
        container.dataframe(summary_p(data),use_container_width=True)
        container.download_button(
        label="Download data as CSV",
        data=summary_p(data).to_csv(index=False),
        file_name='data_analyze.csv',
        mime='text/csv')
    

    container.markdown("---")
    container.markdown("###### Giá trị trung bình (Mean) ######")
    container.markdown("Giá trị trung bình, hay còn gọi là kỳ vọng, là một khái niệm thống kê dùng để đo độ trung tâm của một tập dữ liệu. Nó được tính bằng cách lấy tổng của tất cả các giá trị trong tập dữ liệu và chia cho số lượng các giá trị đó.")
    container.latex(r'''
    \operatorname{E}(X) = \frac{x_1 + x_2 + \dots + x_n}{n}
    ''')

    input1 = st.text_input("Ví dụ giá trị trung bình:",placeholder="Vd:1,2,4,2,5")


    values = input1.split(',')


    numeric_values = []
    for value in values:
        value = value.strip()
        if value:
            numeric_values.append(float(value))


    series1 = pd.Series(numeric_values)


    series1_mean = series1.mean()

    if input1:
        container.markdown(f"Giá trị trung bình của dãy: <span style='color:green;'>{series1_mean}</span>", unsafe_allow_html=True)
    container.markdown("---")

    container.markdown("###### Phương sai (Variance) ######")
    container.markdown("Phương sai (variance) là một thước đo về mức độ phân tán của các giá trị trong một tập dữ liệu. Nó đo lường độ lệch của mỗi giá trị so với giá trị trung bình của tập dữ liệu đó. Phương sai được tính bằng cách lấy tổng bình phương của hiệu giữa mỗi giá trị và giá trị trung bình, chia cho số lượng các giá trị trong tập dữ liệu.")
    container.latex(r'''
    \operatorname{Var}(X) = \sigma^2 = \frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2
    ''')
    container.markdown("Phương sai mẫu hiệu chỉnh được tính bằng cách lấy tổng bình phương của hiệu giữa mỗi giá trị và giá trị trung bình, chia cho số lượng các giá trị trong mẫu trừ 1.")
    container.latex(r'''
    \operatorname{s}^2 = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2
    ''')
    input2 = st.text_input("Ví dụ phương sai:",placeholder="Vd:1,2,4,2,5")
    values2 = input2.split(',')
    numeric_values1 = []
    for value in values2:
        value = value.strip()
        if value:
            numeric_values1.append(float(value))
    series2 = pd.Series(numeric_values1)
    series2_var = series2.var()
    if input1:
        container.markdown(f"Giá trị phương sai hiệu chỉnh của dãy: <span style='color:green;'>{series2_var}</span>", unsafe_allow_html=True)
    container.markdown("---")

    container.markdown("###### Các tứ phân vị (Quartiles) ######")
    container.markdown("Các tứ phân vị (quartiles) là một phương pháp thống kê mô tả để phân chia một tập dữ liệu thành bốn phần bằng nhau. Các tứ phân vị chia tập dữ liệu thành ba khoảng giá trị, được đánh số từ Q1 đến Q3, sao cho khoảng giá trị giữa Q1 và Q3 chứa 50% dữ liệu và khoảng giá trị giữa Q2 (tức là giá trị trung vị) cũng chứa 50% dữ liệu.")
    container.markdown("Tứ phân vị thứ nhất (Q1) tương ứng với phân vị 25%. Nó là giá trị mà có 25% dữ liệu nhỏ hơn hoặc bằng nó và 75% dữ liệu lớn hơn hoặc bằng nó.")
    container.latex(r'''
     \operatorname{Q1} =\begin{cases} x_{(\frac{n+1}{4})}, & \text{n là số chẵn} \\ \frac{1}{2}(x_{(\frac{n+1}{4})} + x_{(\frac{n+3}{4})}), & \text{n là số lẻ} \end{cases}
    ''')
    container.markdown("Tứ phân vị thứ hai (Q2) tương ứng với phân vị 50% hoặc giá trị trung vị. Nó là giá trị mà có 50% dữ liệu nhỏ hơn hoặc bằng nó và 50% dữ liệu lớn hơn hoặc bằng nó.")
    container.latex(r'''
     \operatorname{Q2} = \begin{cases} x_{(\frac{n+1}{2})}, & \text{n là số lẻ} \\ \frac{x_{(\frac{n}{2})} + x_{(\frac{n}{2}+1)}}{2}, & \text{n là số chẵn} \end{cases}
    ''')
    container.markdown("Tứ phân vị thứ ba (Q3) tương ứng với phân vị 75%. Nó là giá trị mà có 75% dữ liệu nhỏ hơn hoặc bằng nó và 25% dữ liệu lớn hơn hoặc bằng nó.")
    container.latex(r'''
     \operatorname{Q3} = \begin{cases} x_{(\frac{3n+1}{4})}, & \text{n là số chẵn} \\ \frac{1}{2}(x_{(\frac{3n+1}{4})} + x_{(\frac{3n+3}{4})}), & \text{n là số lẻ} \end{cases}
    ''')
    container.latex(r'''
    \textcolor{red}{\textbf{Chú ý: }} x_{(i)} \text{ là phần tử thứ $i$}  \text{ của tập dữ liệu đã được sắp xếp theo thứ tự tăng dần.}
    ''')
    #Eg 
    input3 = st.text_input("Ví dụ các tứ phân vị:",placeholder="Vd:1,2,4,2,5")
    values3 = input3.split(',')
    numeric_values3 = []
    for value3 in values3:
        value3 = value3.strip()
        if value3:
            numeric_values3.append(float(value3))
    series3 = pd.Series(numeric_values3)
    q1 = series3.quantile(0.25)
    q2 = series3.quantile(0.5)  # Median
    q3 = series3.quantile(0.75)
    if input3:
        container.markdown(f"Giá trị Q1: <span style='color:green;'>{q1}</span>", unsafe_allow_html=True)
        container.markdown(f"Giá trị Q2 (Trung vị): <span style='color:green;'>{q2}</span>", unsafe_allow_html=True)
        container.markdown(f"Giá trị Q3: <span style='color:green;'>{q3}</span>", unsafe_allow_html=True)
    container.markdown("---")

    
    container.markdown("###### Độ lệch (Skewness) ######")
    container.markdown("Skewness (độ lệch) là một độ đo thống kê được sử dụng để đo mức độ bất đối xứng của phân phối dữ liệu. Nó đo sự chệch lệch của phân phối dữ liệu so với phân phối chuẩn hoặc phân phối đối xứng")
    container.markdown("Nếu phân phối dữ liệu lệch sang phải (có đuôi phân phối dài hơn bên phải so với bên trái), thì giá trị skewness sẽ là số dương. Ngược lại, nếu phân phối dữ liệu lệch sang trái (có đuôi phân phối dài hơn bên trái so với bên phải), thì giá trị skewness sẽ là số âm. Nếu phân phối dữ liệu đối xứng, thì skewness sẽ bằng 0.")
    container.latex(r'''
    \operatorname{S} = \sqrt{n}\frac{\sum_{i=1}^N (x_i - \mu)^3}{\left(\sum_{i=1}^N (x_i - \mu)^2\right)^{3/2}}
    ''')
    container.markdown("Trong trường hợp muốn tính độ lệch skewness mẫu hiệu chỉnh ")
    container.latex(r'''
    \operatorname{S} = \frac{n\sqrt{n-1}}{n-2}\frac{\sum_{i=1}^N (x_i - \mu)^3}{\left(\sum_{i=1}^N (x_i - \mu)^2\right)^{3/2}}
    ''')
    input4 = st.text_input("Ví dụ skewness:",placeholder="Vd:1,2,4,2,5")
    values4 = input4.split(',')
    numeric_values4 = []
    for value4 in values4:
        value4 = value4.strip()
        if value4:
            numeric_values4.append(float(value4))
    series4 = pd.Series(numeric_values4)
    skewness = series4.skew()
    if input4:
        container.markdown(f"Giá trị skewness: <span style='color:green;'>{skewness}</span>", unsafe_allow_html=True)

    container.markdown("---")
    
    container.write("#### Đặc trưng thống kê mẫu nhiều chiều ####")
    container.markdown(
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
    col1,col2 = st.columns(2)
    with col1:
        col1.markdown("###### Ma trận hiệp phương sai ######")
        col1.dataframe(data.cov(),use_container_width=True)
    with col2:
        col2.markdown("######  Ma trận hệ số tương quan ######")
        col2.dataframe(data.corr(),use_container_width=True)
    container.markdown("#### Hiệp phương sai và hệ số tương quan ####")
    container.markdown("Ma trận hiệp phương sai là một ma trận đại diện cho phương sai của các biến ngẫu nhiên trong một tập dữ liệu đa chiều.")
    container.markdown("Ma trận hệ số tương quan là một phiên bản chuẩn hóa của ma trận hiệp phương sai. Nó cũng đại diện cho mối quan hệ giữa các biến trong tập dữ liệu đa chiều, nhưng thay vì hiển thị phương sai, nó hiển thị tương quan giữa các biến. Ma trận này được tính bằng cách chia mỗi phần tử trong ma trận hiệp phương sai cho tích của căn bậc hai của phương sai của hai biến tương ứng.")
    container.markdown("Công thức ma trận hiệp phương sai với phương sai mẫu hiệu chỉnh: ")
    container.latex(r'''
    \mathbf{S} = \frac{1}{n-1}(\mathbf{X}-\boldsymbol{\bar{X}})^T(\mathbf{X}-\boldsymbol{\bar{X}}) = 
        \begin{bmatrix}
            s_{11} & s_{12} & \cdots & s_{1n} \\
            s_{21} & s_{22} & \cdots & s_{2n} \\
            \vdots & \vdots & \ddots & \vdots \\
            s_{n1} & s_{n2} & \cdots & s_{nn}
        \end{bmatrix}

    ''')
    container.latex(r'''
    \text{Trong đó } s_{ij} = \frac{1}{n-1} \sum_{k=1}^{n} (x_{ki} - \bar{x_i})(x_{kj} - \bar{x_j})
    ''')
    container.markdown("Công thức hệ số tương quan: ")
    container.latex(r'''
    r_{XY} = \frac{\operatorname{cov}(X,Y)}{\sigma_X \sigma_Y}
    ''')
    container.markdown("ví dụ hiệp phương sai và hệ số tương quan: ")
    col1, col2 =st.columns(2)
    with col1:
        input5 = st.text_input("Input 1:", placeholder="Example: 1,2,3,4,5")
    with col2:
        input6 = st.text_input("Input 2:", placeholder="Example: 2,4,6,8,10")
    values5 = input5.split(',')
    numeric_values5 = []
    for value5 in values5:
        value5 = value5.strip()
        if value5:
            numeric_values5.append(float(value5))
    
    values6 = input6.split(',')
    numeric_values6 = []
    for value6 in values6:
        value6 = value6.strip()
        if value6:
            numeric_values6.append(float(value6))
    
    series5 = pd.Series(numeric_values5)
    series6 = pd.Series(numeric_values6)
    
    covariance = series5.cov(series6)
    correlation = series5.corr(series6)
    
    if input5 and input6:
        st.markdown(f"Giá trị hiệp phương sai: <span style='color:green;'>{covariance}</span>", unsafe_allow_html=True)
        st.markdown(f"Giá trị hệ số tương quan: <span style='color:green;'>{correlation}</span>", unsafe_allow_html=True)

    container.markdown("---")
    footer()
    image = Image.open("image/sami.jpg")
    with st.sidebar:
        st.sidebar.image(image,width = 50)
        st.sidebar.markdown("# Thống kê #")
        st.sidebar.markdown("---")
        mark_down_text = """
        - Dữ liệu
        - Thống kê mô tả một chiều
            - Bảng thống kê
            - Giá trị trung bình
            - Phương sai
            - Các tứ phân vị
            - Độ lệch Skewness
        - Thống kê mô tả nhiều chiều
            - Hiệp phương sai
            - Hệ số tương quan
        """
        st.sidebar.markdown(mark_down_text)

#### Data viusualyzation
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

    elif chart_type == "Line":
        st.header("Line Chart")
        multiple = st.checkbox("Vẽ nhiều đường", value=False)
        col1,col2 = st.columns(2)
        with col1:
            x_column = st.selectbox("Chọn trục X", data.columns)

        if multiple:
            container = st.empty()
            number = container.number_input("Nhập số lượng đường", min_value=1, step=1, value=1)
            with col2:
                y_columns = []
                for i in range(number):
                    with col2:
                        y_column = st.selectbox(f"Chọn trục Y {i+1}", data.columns)
                        y_columns.append(y_column)

    # Create line chart with multiple lines
            fig = px.line(data, x=x_column, y=y_columns, markers=True)

        else:
            with col2:
                y_column = st.selectbox("Chọn trục Y", data.columns)

        # Create line chart with single line
            fig = px.line(data, x=x_column, y=y_column, markers=True)

        st.plotly_chart(fig,theme=None, use_container_width=True)

    elif chart_type == "Scatter":

        st.header("Scatter Chart")
        col1,col2 = st.columns(2)
        with col1:
            x_column = st.selectbox("Chọn trục X", data.columns)
        with col2:
            y_column = st.selectbox("Chọn trục Y", data.columns)
        fig = px.scatter(data, x=x_column, y=y_column,color=x_column)
        st.plotly_chart(fig,theme=None, use_container_width=True)

    elif chart_type == "Pie":

        st.header("Biểu đồ tròn")
        col1,col2 = st.columns(2)
        with col1:
            x_column = st.selectbox("Chọn nhãn", data.columns)
        with col2:
            y_column = st.selectbox("Chọn giá trị", data.columns)
        donut = st.checkbox('Sử dụng donut', value=True)
        if donut:
        # compute and show the sample statistics
            hole = 0.4
   
        else:
         # compute and show the population statistics
            hole = 0
        fig = px.pie(data,names = x_column,values = y_column,hole=hole)
        st.plotly_chart(fig,theme=None, use_container_width=True)
    
    elif chart_type == "Boxplot":
        st.header("Biểu đồ Hộp")
        col1,col2 = st.columns(2)
        with col1:
            x_column = st.selectbox("Chọn trục X", data.columns)
        with col2:
            y_column = st.selectbox("Chọn trục Y", data.columns)

        fig = px.box(data,x = x_column,y = y_column, )
        st.plotly_chart(fig,theme=None, use_container_width=True)
    image = Image.open("image/sami.jpg")
    with st.sidebar:
        st.sidebar.image(image,width = 50)
        st.sidebar.markdown("# Trực quan hóa #")
        st.sidebar.markdown("---")
        mark_down_text = """
        """
        st.sidebar.markdown(mark_down_text)


#### hypothesis test
def hypothesis_test(test_type, data):
    # Perform basic data analysis
    
    ######
    if test_type =="Kiểm định một mẫu":
        container.write("#### Kiểm định về giá trị trung bình ####")
        numeric_columns = data.select_dtypes(include=["int", "float"]).columns
        x_column = st.selectbox("Chọn cột cần kiểm định ", numeric_columns)
        container.markdown("Các yếu tố: ")
        col1, col2, col3 = st.columns(3)
        with col1:
            clevel = st.text_input('Mức ý nghĩa', '0.05')
        with col2:
            a0 = st.text_input('Giá trị cần kiểm định', '')

        with col3:
            H1 = st.selectbox("Đối thuyết", ["Khác", "Lớn hơn", "Nhỏ hơn"])
        
        sample = data[x_column].values
        alpha = float(clevel)
        container.markdown("---")   


        if a0.strip():  # Check if a0 is not empty or whitespace
            container.markdown("###### Bài toán kiểm định giả thuyết:")
            col1, col2, col3 = st.columns(3)
            with col2:
                if H1 == "Khác":
                    st.latex(r'''
                    \left\{
                    \begin{aligned}
                        H_0 &= \mu \\
                        H_1 &\neq \mu
                    \end{aligned}
                    \right.
                    ''')
                elif H1 == "Lớn hơn":
                    st.latex(r'''
                    \left\{
                    \begin{aligned}
                        H_0 &= \mu \\
                        H_1 &> \mu
                    \end{aligned}
                    \right.
                    ''')
                else:
                    st.latex(r'''
                    \left\{
                    \begin{aligned}
                        H_0 &= \mu \\
                        H_1 &< \mu
                    \end{aligned}
                    \right.
                    ''')
            stats_df = pd.DataFrame({
            "Mean": [data[x_column].mean()],
            "Standard Deviation": [data[x_column].std()],
            "Count": [data[x_column].count()]
        })
            
            container.markdown("Giá trị thống kê tính được")
            reset_df=stats_df.set_index("Mean",drop=True)
            container.dataframe(reset_df,use_container_width=True)
            a0_value = float(a0)
            container.markdown("Thống kê phù hợp t:")
            container.latex(r'''
            t=\dfrac{(\overline{x}-\mu)\sqrt{n}}{s_d}
            ''')

            if H1 == "Khác":
                t_statistic, p_value= stats.ttest_1samp(sample, popmean=a0_value)
                st.markdown(f"t-statistic= :green[{t_statistic}]")
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
                st.plotly_chart(fig,theme=None, use_container_width=True)
                
                sigma = (data[x_column].std())/math.sqrt(data[x_column].count())  # Sample stdev/sample size

                interval=stats.t.interval(1-alpha,                        # Confidence level
                    df = data[x_column].count()-1,                     # Degrees of freedom
                    loc = data[x_column].mean(), # Sample mean
                    scale= sigma)
                st.markdown(f"Khoảng tin cậy 2 phía: :green[{interval}]" )
                st.markdown("##### Kết luận")
                if(np.abs(t_statistic) > percent):
                    latex_expression = r"t_{n-1}(\frac{\alpha}{2})"
                    st.markdown(f"Vì |t_statistic| = :green[{np.abs(t_statistic)}] > $$ {latex_expression}$$ = :green[{percent}] ")
                    st.markdown(f"nên ta bác bỏ giả thuyết H0 ở mức ý nghĩa :green[{alpha}]")
                else:
                    latex_expression = r"t_{n-1}(\frac{\alpha}{2})"
                    st.markdown(f"Vì |t_statistic|= :green[{np.abs(t_statistic)}] < $$ {latex_expression}$$=:green[{percent}] ")
                    st.markdown(f"nên ta chấp nhận giả thuyết H0 ở mức ý nghĩa :green[{alpha}]")

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


                st.plotly_chart(fig, theme=None, use_container_width=True)
                st.markdown("##### Kết luận")
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

                st.plotly_chart(fig, theme=None, use_container_width=True)
                st.markdown("##### Kết luận")
                if(t_statistic < percent):
                    latex_expression = r"t_{n-1}({\alpha})"
                    st.markdown(f"Vì t_statistic= :green[{t_statistic}] < $$ {latex_expression}$$ = :green[{percent}] ")
                    st.markdown(f"nên ta bác bỏ giả thuyết H0 ở mức ý nghĩa :green[{alpha}]")
                else:
                    latex_expression = r"t_{n-1}({\alpha})"
                    st.markdown(f"Vì t_statistic= :green[{t_statistic}] > $$ {latex_expression}$$=:green[{percent}] ")
                    st.markdown(f"nên ta chấp nhận giả thuyết H0 ở mức ý nghĩa :green[{alpha}]")  

        #
        container.write("#### Kiểm định về phương sai ####")
        numeric_columns = data.select_dtypes(include=["int", "float"]).columns
        x_column = st.selectbox("Chọn cột cần kiểm định 1 ", numeric_columns)
        container.markdown("Các yếu tố: ")
        col1, col2 = st.columns(2)
        with col1:
            clevel = st.text_input('Mức ý nghĩa1', '0.05')
        with col2:
            a0 = st.text_input('Giá trị cần kiểm định1', '')

        sample = data[x_column].values
        alpha = float(clevel)
        container.markdown("---")   

        if a0.strip():  # Check if a0 is not empty or whitespace
            container.markdown("###### Bài toán kiểm định giả thuyết:")
            col1, col2 = st.columns(2)
            with col2:
                st.latex(r'''
                \left\{
                \begin{aligned}
                    H_0 &: \sigma^2 = \sigma_0^2 \\
                    H_1 &: \sigma^2 \neq \sigma_0^2
                \end{aligned}
                \right.
                ''')

            stats_df = pd.DataFrame({
                "Variance": [data[x_column].var()],
                "Count": [data[x_column].count()]
            })
                
            container.markdown("Giá trị thống kê tính được")
            reset_df = stats_df.set_index("Variance", drop=True)
            container.dataframe(reset_df, use_container_width=True)

            a0_value = float(a0)
            container.markdown("Thống kê phù hợp chi-square:")
            container.latex(r'''
            \chi^2 = (n-1) \cdot \frac{{s^2}}{{\sigma_0^2}}
            ''')

            chi2_statistic = (data[x_column].count() - 1) * data[x_column].var() / a0_value
            st.markdown(f"chi-square statistic = :green[{chi2_statistic}]")
            chi2_critical = stats.chi2.ppf(1 - alpha / 2, df=data[x_column].count() - 1)

            # Generate x values for the Chi-square distribution plot
            x = np.linspace(0, 20, 1000)

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

            x_fill1 = np.linspace(0, chi2_critical, 1000)
            pdf_fill1 = stats.chi2.pdf(x_fill1, df=data[x_column].count()-1)

            x_fill2 = np.linspace(chi2_critical, x[-1], 1000)
            pdf_fill2 = stats.chi2.pdf(x_fill2, df=data[x_column].count()-1)

            fig.add_trace(go.Scatter(x=x_fill1, y=pdf_fill1, name="Rejection Region", fill='tozeroy'))
            fig.add_trace(go.Scatter(x=x_fill2, y=pdf_fill2, name="Rejection Region", fill='tozeroy'))
            
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            if chi2_statistic > chi2_critical :
                container.markdown(":red[Không chấp nhận null hypothesis]")
                container.markdown("Có bằng chứng đủ để bác bỏ giả thuyết H0.")
            else:
                container.markdown(":green[Chấp nhận null hypothesis]")
                container.markdown("Không có bằng chứng đủ để bác bỏ giả thuyết H0.")
        
        
    image = Image.open("image/sami.jpg")
    with st.sidebar:
        st.sidebar.image(image,width = 50)
        st.sidebar.markdown("# Thống kê #")
        st.sidebar.markdown("---")
        mark_down_text = """
        - Dữ liệu
        - Thống kê mô tả một chiều
            - Bảng thống kê
            - Giá trị trung bình
            - Phương sai
            - Các tứ phân vị
            - Độ lệch Skewness
        - Thống kê mô tả nhiều chiều
            - Hiệp phương sai
            - Hệ số tương quan
        """
        st.sidebar.markdown(mark_down_text)

        
# main function
def main():

    
    image = Image.open("image/sami.jpg")
    
    container.image(image,width = 100)
    container.write(" # Thống kê và phân tích dữ liệu # ")

    with container:

        selected = option_menu(None, ["Dữ liệu", "Thống kê", "Trực quan hóa","Kiểm định"], 
            icons=['clipboard-data', 'table', "bar-chart-fill", 'clipboard-check'], 
            menu_icon="cast", default_index=0, orientation="horizontal")
        
        container.markdown(
                """
                <style>
                .b {
                    margin-top: 50px ;
                    }
                </style>

                <div class="b"></div>
                """,
                unsafe_allow_html=True
            )

    
        container.markdown("### Tải lên dữ liệu ###")
        file = st.file_uploader("",type=["csv"])


        if file is not None:

            data = load_data(file)

            if selected =='Dữ liệu':
                info(data)

            if selected == 'Thống kê':
                analyze_data(data)

            if selected =='Trực quan hóa':

                container.write(" # Trực quan hóa dữ liệu # ")
                container.write("#### Dữ liệu ####")
                container.write("Data")
                container.dataframe(data,use_container_width=True)
                container.markdown("---")

                chart_type = st.selectbox("Chọn loại biểu đồ", ["Bar", "Line", "Scatter","Pie","Boxplot"])

                create_chart(chart_type, data)

            if selected =='Kiểm định':
                container.write(" # Kiểm định giả thuyết thống kê # ")
                container.write("#### Dữ liệu ####")
                container.write("Data")
                container.dataframe(data,use_container_width=True)
                container.markdown("---")
                test_type = st.selectbox("Chọn phương thức muốn kiểm định", ["Kiểm định một mẫu", "So sánh hai mẫu", "Kiểm định phi tham số"])
                hypothesis_test(test_type, data)
        else:
            st.sidebar.image(image,width = 50)
            st.sidebar.markdown("---")

            container.markdown("---")
            
            st.markdown(
                """
                <style>
                .b {
                    margin-top: 50px ;
                    }
                </style>

                <div class="b"></div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(" ### Làm sao để sử dụng ?")
            st.markdown(
                """
                <style>
                .b {
                    margin-top: 50px ;
                    }
                </style>

                <div class="b"></div>
                """,
                unsafe_allow_html=True
            )

            col1,col2=st.columns(2)
            with col1:
                st.markdown("""
                        <head>
                        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                        </head>
                        <body>

                        <i class="fa-solid fa-1 fa-beat" style="font-size:70px;color: #ff4b4b;"></i>
                        <h5>Tải lên dữ liệu của bạn</h5>
                        </body>
                                            
                        
                        """, unsafe_allow_html=True)
                image1 = Image.open("image/im1.png")
                st.image(image1)

                st.markdown(
                """
                <style>
                .b {
                    margin-top: 50px ;
                    }
                </style>

                <div class="b"></div>
                """,
                unsafe_allow_html=True
            )
                st.markdown("""
                        <head>
                        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                        </head>
                        <body>

                        <i class="fa-solid fa-3 fa-beat" style="font-size:70px;color: #ff4b4b;"></i>
                        <h5>Bắt đầu tính toán </h5>
                        </body>
                                            
                        
                        """, unsafe_allow_html=True)
                image3 = Image.open("image/im3.png")
                st.image(image3)
                
            with col2:
                st.markdown("""
                        <head>
                        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                        </head>
                        <body>

                        <i class="fa-solid fa-2 fa-beat" style="font-size:70px;color: #ff4b4b;"></i>
                        <h5>Chọn chức năng mong muốn</h5>
                        </body>
                                            
                        
                        """, unsafe_allow_html=True)
                image2 = Image.open("image/im2.png")
                st.image(image2)
                st.markdown(
                """
                <style>
                .b {
                    margin-top: 50px ;
                    }
                </style>

                <div class="b"></div>
                """,
                unsafe_allow_html=True
            )
                st.markdown("""
                        <head>
                        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                        </head>
                        <body>

                        <i class="fa-solid fa-4 fa-beat" style="font-size:70px;color: #ff4b4b;"></i>
                        <h5>Tải xuống và tiếp tục công việc</h5>
                        </body>
                                            
                        
                        """, unsafe_allow_html=True)
                image4 = Image.open("image/im4.png")
                st.image(image4)
            container.markdown("---")
            footer()
    
    
if __name__ == "__main__":
    main()
    

    

    

