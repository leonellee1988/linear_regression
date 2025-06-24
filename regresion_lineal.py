import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Linear Regression Model", layout="centered")
st.title("📊 Interactive Linear Regression Model")

# Initialize session state variables
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False
if 'show_correlation' not in st.session_state:
    st.session_state.show_correlation = False

# URL input
st.markdown("#### 🔗 Enter your dataset URL:")
url = st.text_input(
    label="", 
    placeholder="Example: https://.../file.csv", 
    label_visibility="collapsed"
)

if st.button("📥 Load") and url:
    try:
        if url.endswith('.csv'):
            df = pd.read_csv(url)
        elif url.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(url)
        else:
            st.error("Unsupported format. Use CSV or Excel.")
            df = None
        if df is not None:
            st.session_state.df = df
            st.session_state.dataset_loaded = True
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.session_state.dataset_loaded = False

# If dataset is loaded
if st.session_state.dataset_loaded:
    df = st.session_state.df

    st.subheader("📋 Dataset Overview")
    st.write(f"🔢 Rows: {df.shape[0]}")
    st.write(f"📁 Columns: {df.shape[1]}")
    st.dataframe(df.head())

    st.subheader("📚 Column Types")
    dtypes = df.dtypes.reset_index()
    dtypes.columns = ['Column', 'Type']
    st.dataframe(dtypes)

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    st.subheader("🎯 Variable Selection")
    selected_target = st.selectbox(
        "Select the dependent variable (Y):",
        options=numeric_cols,
        index=None,
        placeholder="Select a variable"
    )

    selected_features = st.multiselect(
        "Select independent attributes (X): (Max. 3)",
        options=[col for col in numeric_cols if col != selected_target],
        max_selections=3
    )

    if st.button("📊 Generate correlation and pairplot"):
        if selected_target and selected_features:
            st.session_state.features = selected_features
            st.session_state.target = selected_target
            st.session_state.show_correlation = True
        else:
            st.warning("You must select a dependent variable and at least one attribute.")

    # Show correlation and pairplot if already defined
    if st.session_state.show_correlation and "features" in st.session_state and "target" in st.session_state:
        st.subheader("📈 Correlation Matrix")
        selected = st.session_state.features + [st.session_state.target]
        corr_matrix = df[selected].corr()
        st.dataframe(corr_matrix)
        fig1, ax1 = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax1)
        st.pyplot(fig1)

        st.subheader("🔗 Pairplot")
        pair_fig = sns.pairplot(df[selected]).fig
        st.pyplot(pair_fig)

    if "features" in st.session_state and "target" in st.session_state:
        features = st.session_state.features
        target = st.session_state.target

        st.subheader("⚙️ Model Hyperparameters")
        learning_rate = st.number_input(
            "Learning rate",
            min_value=0.0001,
            max_value=1.0,
            step=0.0001,
            format="%.4f",
            value=None,
            placeholder="E.g.: 0.01"
        )
        epochs = st.number_input("Epochs (max. 500)", min_value=1, max_value=500, step=1, value=None)
        max_batch_size = max(1, df.shape[0] // 2)
        batch_size = st.number_input("Batch size", min_value=1, max_value=max_batch_size, step=1, value=None)

        if st.button("🚀 Train Model"):
            if None in (learning_rate, epochs, batch_size):
                st.error("All hyperparameters must be entered.")
            else:
                X = df[features].values
                y = df[target].values.reshape(-1, 1)

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

                weights = np.zeros((X_train.shape[1], 1))
                bias = 0.0
                rmse_list = []

                for epoch in range(int(epochs)):
                    indices = np.random.permutation(X_train.shape[0])
                    X_shuffled = X_train[indices]
                    y_shuffled = y_train[indices]

                    for i in range(0, X_train.shape[0], batch_size):
                        X_batch = X_shuffled[i:i+batch_size]
                        y_batch = y_shuffled[i:i+batch_size]
                        y_pred = X_batch @ weights + bias
                        error = y_pred - y_batch
                        dw = (X_batch.T @ error) / len(y_batch)
                        db = np.mean(error)
                        weights -= learning_rate * dw
                        bias -= learning_rate * db

                    y_val_pred = X_val @ weights + bias
                    rmse = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
                    rmse_list.append(rmse)

                best_epoch = np.argmin(rmse_list) + 1
                best_rmse = rmse_list[best_epoch - 1]
                rmse_improvement = rmse_list[-2] - rmse_list[-1] if len(rmse_list) > 1 else 0

                st.subheader("📉 Loss Curve (RMSE per Epoch)")
                fig2, ax2 = plt.subplots()
                ax2.plot(rmse_list)
                ax2.set_xlabel("Epochs")
                ax2.set_ylabel("RMSE")
                st.pyplot(fig2)

                st.subheader("📈 Generated Model")
                formula = f"Y = {round(bias, 4)}"
                for coef, name in zip(weights.flatten(), features):
                    formula += f" + ({round(coef, 4)} × {name})"
                st.markdown(f"**{formula}**")

                st.markdown(f"✅ Best model achieved at epoch **{best_epoch}** with RMSE of **{round(best_rmse, 4)}**")
                st.markdown(f"📉 Final RMSE improvement over previous epoch: **{round(rmse_improvement, 8)}**")

                if len(features) == 1:
                    fig3, ax3 = plt.subplots()
                    ax3.scatter(X[:, 0], y, label='Actual data')
                    pred_line = X_scaled[:, 0].reshape(-1, 1) @ weights + bias
                    ax3.plot(X[:, 0], pred_line, color='red', label='Regression line')
                    ax3.set_xlabel(features[0])
                    ax3.set_ylabel(target)
                    ax3.legend()
                    st.pyplot(fig3)

                st.subheader("🔍 Model Validation")
                df_pred = pd.DataFrame({
                    "Actual": y_val.flatten(),
                    "Predicted": y_val_pred.flatten()
                })
                st.dataframe(df_pred.head(10))
                st.write(f"📌 Final RMSE: {round(rmse_list[-1], 4)}")

st.markdown("""
<hr style='border:1px solid #ddd; margin-top: 40px; margin-bottom:10px'>
<div style='text-align: center; color: grey; font-size: 0.9em'>
    Developed by Edwin Lee | 📧 leonellee2016@gmail.com | Github: leonellee1988
</div>
""", unsafe_allow_html=True)